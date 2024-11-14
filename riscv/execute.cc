// See LICENSE for license details.

#include "processor.h"
#include "mmu.h"
#include "disasm.h"
#include <iomanip>
#include <cstring>
#include <cassert>
#include <functional>

#ifdef RISCV_ENABLE_COMMITLOG
static void commit_log_reset(processor_t* p)
{
  p->get_state()->log_reg_write.clear();
  p->get_state()->log_mem_read.clear();
  p->get_state()->log_mem_write.clear();
}

static void commit_log_stash_privilege(processor_t* p)
{
  state_t* state = p->get_state();
  state->last_inst_priv = state->prv;
  state->last_inst_xlen = p->get_xlen();
  state->last_inst_flen = p->get_flen();
}

static void commit_log_print_value(FILE *log_file, int width, const void *data)
{
  assert(log_file);

  switch (width) {
    case 8:
      fprintf(log_file, "0x%01" PRIx8, *(const uint8_t *)data);
      break;
    case 16:
      fprintf(log_file, "0x%04" PRIx16, *(const uint16_t *)data);
      break;
    case 32:
      fprintf(log_file, "0x%08" PRIx32, *(const uint32_t *)data);
      break;
    case 64:
      fprintf(log_file, "0x%016" PRIx64, *(const uint64_t *)data);
      break;
    default:
      // max lengh of vector
      if (((width - 1) & width) == 0) {
        const uint64_t *arr = (const uint64_t *)data;

        fprintf(log_file, "0x");
        for (int idx = width / 64 - 1; idx >= 0; --idx) {
          fprintf(log_file, "%016" PRIx64, arr[idx]);
        }
      } else {
        abort();
      }
      break;
  }
}

static void commit_log_print_value(FILE *log_file, int width, uint64_t val)
{
  commit_log_print_value(log_file, width, &val);
}

const char* processor_t::get_symbol(uint64_t addr)
{
  return sim->get_symbol(addr);
}

static void commit_log_print_insn(processor_t *p, reg_t pc, insn_t insn)
{
  FILE *log_file = p->get_log_file();

  auto& reg = p->get_state()->log_reg_write;
  auto& load = p->get_state()->log_mem_read;
  auto& store = p->get_state()->log_mem_write;
  int priv = p->get_state()->last_inst_priv;
  int xlen = p->get_state()->last_inst_xlen;
  int flen = p->get_state()->last_inst_flen;

  // print core id on all lines so it is easy to grep
  fprintf(log_file, "core%4" PRId32 ": ", p->get_id());

  fprintf(log_file, "%1d ", priv);
  commit_log_print_value(log_file, xlen, pc);
  fprintf(log_file, " (");
  commit_log_print_value(log_file, insn.length() * 8, insn.bits());
  fprintf(log_file, ")");
  bool show_vec = false;

  for (auto item : reg) {
    if (item.first == 0)
      continue;

    char prefix;
    int size;
    int rd = item.first >> 4;
    bool is_vec = false;
    bool is_vreg = false;
    switch (item.first & 0xf) {
    case 0:
      size = xlen;
      prefix = 'x';
      break;
    case 1:
      size = flen;
      prefix = 'f';
      break;
    case 2:
      size = p->VU.VLEN;
      prefix = 'v';
      is_vreg = true;
      break;
    case 3:
      is_vec = true;
      break;
    case 4:
      size = xlen;
      prefix = 'c';
      break;
    default:
      assert("can't been here" && 0);
      break;
    }

    if (!show_vec && (is_vreg || is_vec)) {
        fprintf(log_file, " e%ld %s%ld l%ld",
                p->VU.vsew,
                p->VU.vflmul < 1 ? "mf" : "m",
                p->VU.vflmul < 1 ? (reg_t)(1 / p->VU.vflmul) : (reg_t)p->VU.vflmul,
                p->VU.vl->read());
        show_vec = true;
    }

    if (!is_vec) {
      if (prefix == 'c')
        fprintf(log_file, " c%d_%s ", rd, csr_name(rd));
      else
        fprintf(log_file, " %c%2d ", prefix, rd);
      if (is_vreg)
        commit_log_print_value(log_file, size, &p->VU.elt<uint8_t>(rd, 0));
      else
        commit_log_print_value(log_file, size, item.second.v);
    }
  }

  for (auto item : load) {
    fprintf(log_file, " mem ");
    commit_log_print_value(log_file, xlen, std::get<0>(item));
  }

  for (auto item : store) {
    fprintf(log_file, " mem ");
    commit_log_print_value(log_file, xlen, std::get<0>(item));
    fprintf(log_file, " ");
    commit_log_print_value(log_file, std::get<2>(item) << 3, std::get<1>(item));
  }
  fprintf(log_file, "\n");
}
#else
static void commit_log_reset(processor_t* p) {}
static void commit_log_stash_privilege(processor_t* p) {}
static void commit_log_print_insn(processor_t* p, reg_t pc, insn_t insn) {}
#endif

inline void processor_t::update_histogram(reg_t pc)
{
#ifdef RISCV_ENABLE_HISTOGRAM
  pc_histogram[pc]++;
#endif
}

// This is expected to be inlined by the compiler so each use of execute_insn
// includes a duplicated body of the function to get separate fetch.func
// function calls.
static inline reg_t execute_insn(processor_t* p, reg_t pc, insn_fetch_t fetch)
{
  commit_log_reset(p);
  commit_log_stash_privilege(p);
  reg_t npc;

  try {
    npc = fetch.func(p, fetch.insn, pc);
    if (npc != PC_SERIALIZE_BEFORE) {

#ifdef RISCV_ENABLE_COMMITLOG
      if (p->get_log_commits_enabled()) {
        commit_log_print_insn(p, pc, fetch.insn);
      }
#endif

     }
#ifdef RISCV_ENABLE_COMMITLOG
  } catch (wait_for_interrupt_t &t) {
      if (p->get_log_commits_enabled()) {
        commit_log_print_insn(p, pc, fetch.insn);
      }
      throw;
  } catch(mem_trap_t& t) {
      //handle segfault in midlle of vector load/store
      if (p->get_log_commits_enabled()) {
        for (auto item : p->get_state()->log_reg_write) {
          if ((item.first & 3) == 3) {
            commit_log_print_insn(p, pc, fetch.insn);
            break;
          }
        }
      }
      throw;
#endif
  } catch(...) {
    throw;
  }
  p->update_histogram(pc);

  return npc;
}

bool processor_t::slow_path()
{
  return debug || state.single_step != state.STEP_NONE || state.debug_mode;
}

// fetch/decode/execute loop
void processor_t::step(size_t n, long long* p_cycle)
{
  if (!state.debug_mode) {
    if (halt_request == HR_REGULAR) {
      enter_debug_mode(DCSR_CAUSE_DEBUGINT);
    } else if (halt_request == HR_GROUP) {
      enter_debug_mode(DCSR_CAUSE_GROUP);
    } // !!!The halt bit in DCSR is deprecated.
    else if (state.dcsr->halt) {
      enter_debug_mode(DCSR_CAUSE_HALT);
    }
  }

  while (n > 0) {
    size_t instret = 0;  // Instructions Retired(실행 완료 후 처리한 명령어 수)로 추정
    reg_t pc = state.pc; // 현재 pc 레지스터 값
    mmu_t* _mmu = mmu;

    #define advance_pc() \
     if (unlikely(invalid_pc(pc))) { \
       switch (pc) { \
         case PC_SERIALIZE_BEFORE: state.serialized = true; break; \
         case PC_SERIALIZE_AFTER: ++instret; break; \
         case PC_SERIALIZE_WFI: n = ++instret; break; \
         default: abort(); \
       } \
       pc = state.pc; \
       break; \
     } else { \
       state.pc = pc; \
       instret++; \
     }

    try
    {
      take_pending_interrupt();

      // Main simulation loop, slow path.
      while (instret < n)
      {
        if (unlikely(!state.serialized && state.single_step == state.STEP_STEPPED)) {
          state.single_step = state.STEP_NONE;
          if (!state.debug_mode) {
            enter_debug_mode(DCSR_CAUSE_STEP);
            // enter_debug_mode changed state.pc, so we can't just continue.
            break;
          }
        }

        if (unlikely(state.single_step == state.STEP_STEPPING)) {
          state.single_step = state.STEP_STEPPED;
        }

        /* Exit main function */
        if (pc == 0x000000000001017C) {
          std::cerr << "\033[33m" << "Exit main!!" << "\033[0m" << std::endl;
          main_inside = false;
        }
        /* ------------------ */

        insn_fetch_t fetch = mmu->load_insn(pc); // pc 주소의 instruction 가져오기
        if (debug && !state.serialized)
          disasm(fetch.insn); // -l 옵션 사용 시

        /* Fetch Insn */
        reg_t insn_pc = pc;
        insn_t insn = fetch.insn;
        insn_bits_t insn_bits = insn.bits();
        insn_bits_t opcode = insn_bits & 0x7f; // 0x7f = 0b01111111
        insn_bits_t funct7 = (insn_bits >> 25) & 0x7f;
        if (main_inside && !trap_inside) {
          std::cerr
              << "\033[90m"
              << "(PC: " << "0x" << std::setw(16) << std::setfill('0') << std::hex << insn_pc << ") "
              << "\033[0m"
              << "0x" << std::setw(8) << std::setfill('0') << std::hex << insn_bits; // for debugging
        }
        /* ---------- */

        pc = execute_insn(this, pc, fetch); // instruction 실행

        /* Push Nop lamda function */
        auto pushNop = [this](size_t idx = 0) {
          if (insn_buf.size() >= 5)
            insn_buf.pop_back();
          insn_buf.insert(insn_buf.begin() + idx, 0x00000013);
        };
        /* ----------------------- */

        if (main_inside) {
          /* Calculate Pipeline Cycle */
          if (!trap_inside && !page_fault) {
            if (insn_buf.size() >= 5)
              insn_buf.pop_back();
            insn_buf.push_front(insn);

            /* Check Pipeline Stall */
            if ((opcode == 0b0110011 || opcode == 0b0111011) && funct7 != 1) { // R-type
              std::cerr << ": R-type" << std::endl;
              // Assume that insn in a EX stage
              uint64_t rs1 = insn.rs1();
              uint64_t rs2 = insn.rs2();

              // Get rd register of an insn in a MEM stage
              if (insn_buf.size() >= 2) {
                insn_bits_t opcode_MEM = insn_buf[1].bits() & 0x7f;
                uint64_t rd = insn_buf[1].rd();
                if (opcode_MEM == 0b0000011 && rd != 0 && (rd == rs1 || rd == rs2)) { // check load
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
              }
            }
            else if (opcode == 0b0010011 || opcode == 0b0011011 || opcode == 0b0000011) { // I-type
              std::cerr << ": I-type" << std::endl;
              // Assume that insn in a EX stage
              uint64_t rs1 = insn.rs1();

              // Get rd register of an insn in a MEM stage
              if (insn_buf.size() >= 2) {
                insn_bits_t opcode_MEM = insn_buf[1].bits() & 0x7f;
                uint64_t rd = insn_buf[1].rd();
                if (opcode_MEM == 0b0000011 && rd != 0 && rd == rs1) { // check load
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
              }
            }
            else if (opcode == 0b0100011) { // S-type
              std::cerr << ": S-type" << std::endl;
              // Assume that insn in a EX stage
              uint64_t rs1 = insn.rs1();

              // Get rd register of an insn in a MEM stage
              if (insn_buf.size() >= 2) {
                insn_bits_t opcode_MEM = insn_buf[1].bits() & 0x7f;
                uint64_t rd = insn_buf[1].rd();
                if (opcode_MEM == 0b0000011 && rd != 0 && rd == rs1) { // check load
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
              }
            }
            else if (opcode == 0b1100011) { // B-type
              std::cerr << ": B-type" << std::endl;
              // Assume that insn in a ID stage
              uint64_t rs1 = insn.rs1();
              uint64_t rs2 = insn.rs2();

              // Get rd register of an insn in a EX stage
              if (insn_buf.size() >= 2) {
                insn_bits_t opcode_EX = insn_buf[1].bits() & 0x7f;
                uint64_t rd = insn_buf[1].rd();
                if ((((opcode_EX == 0b0110011 || opcode_EX == 0b0111011) && funct7 != 1) || // check R-type
                    opcode_EX == 0b0010011 || opcode_EX == 0b0011011 || // check I-type(non-load)
                    opcode_EX == 0b0110111 || opcode_EX == 0b0010111) && // check U-type
                    rd != 0 && (rd == rs1 || rd == rs2)) {
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
                else if (opcode_EX == 0b0000011 && rd != 0 && (rd == rs1 || rd == rs2)) { // check load
                  pushNop(1);
                  pushNop(1); // bubble
                  (*p_cycle) += 2;
                }
              }

              // Get rd register of an insn in a MEM stage
              if (insn_buf.size() >= 3) {
                insn_bits_t opcode_MEM = insn_buf[2].bits() & 0x7f;
                uint64_t rd = insn_buf[2].rd();
                if (opcode_MEM == 0b0000011 && rd != 0 && (rd == rs1 || rd == rs2)) { // check load
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
              }

              if (insn_pc + 4 != pc) { // branch taken
                pushNop(); // bubble
                (*p_cycle)++;
              }
            }
            else if (opcode == 0b0110111 || opcode == 0b0010111) { // U-type
              std::cerr << ": U-type" << std::endl;
            }
            else if (opcode == 0b1101111) { // JAL
              std::cerr << ": JAL" << std::endl;
              pushNop(); // bubble
              (*p_cycle)++;
            }
            else if (opcode == 0b1100111) { // JALR
              std::cerr << ": JALR" << std::endl;
              // Assume that insn in a ID stage
              uint64_t rs1 = insn.rs1();

              // Get rd register of an insn in a EX stage
              if (insn_buf.size() >= 2) {
                insn_bits_t opcode_EX = insn_buf[1].bits() & 0x7f;
                uint64_t rd = insn_buf[1].rd();
                if ((((opcode_EX == 0b0110011 || opcode_EX == 0b0111011) && funct7 != 1) || // check R-type
                    opcode_EX == 0b0010011 || opcode_EX == 0b0011011 || // check I-type(non-load)
                    opcode_EX == 0b0110111 || opcode_EX == 0b0010111) && // check U-type
                    rd != 0 && rd == rs1) {
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
                else if (opcode_EX == 0b0000011 && rd != 0 && rd == rs1) { // check load
                  pushNop(1);
                  pushNop(1); // bubble
                  (*p_cycle) += 2;
                }
              }

              // Get rd register of an insn in a MEM stage
              if (insn_buf.size() >= 3) {
                insn_bits_t opcode_MEM = insn_buf[2].bits() & 0x7f;
                uint64_t rd = insn_buf[2].rd();
                if (opcode_MEM == 0b0000011 && rd != 0 && rd == rs1) { // check load
                  pushNop(1); // bubble
                  (*p_cycle)++;
                }
              }

              pushNop(); // bubble
              (*p_cycle)++;
            }
            else { // RV32I 또는 RV64I에 속하지 않음
              std::cerr << ": nop" << std::endl;
              insn_buf.pop_front();
              insn_buf.push_front(0x00000013); // nop
            }
            /* -------------------- */

            (*p_cycle)++;
          }
          else if (!trap_inside && page_fault) { // handle instruction duplicate
            std::cerr << "\033[32m" << ": Insn dup" << "\033[0m" << std::endl;
            page_fault = false;
          }
          /* ------------------------ */

          /* Detect sret || mret */
          if (insn_bits == 0x10200073 || insn_bits == 0x30200073) {
            std::cerr << "\033[32m" << "sret / mret" << "\033[0m" << std::endl;
            trap_inside--;
          }
          /* ------------------- */
        }

        /* Jump to main function */
        if (pc == 0x0000000000010178) {
          std::cerr << "\033[33m" << "Jump to main!!" << "\033[0m" << std::endl;
          main_inside = true;
        }
        /* --------------------- */

        advance_pc();
      }
    }
    catch(trap_t& t) // trap 발생
    {
      if (main_inside) {
        std::cerr << std::endl << "\033[31m" << "<< Trap: "; // for debugging
        trap_inside++;

        /* Detect ecall and page fault */
        if (t.cause() == 8) // user mode ecall
          std::cerr << "ecall >>" << "\033[0m" << std::endl;    // for debugging

        else if (t.cause() == 13 || t.cause() == 15) { // page fault by L/S
          std::cerr << "Page Fault >>" << "\033[0m" << std::endl; // for debugging
          if (trap_inside == 1)
            page_fault = true;
        }

        else
          std::cerr << ">>" << "\033[0m" << std::endl;
        /* --------------------------- */
      }

      take_trap(t, pc);
      n = instret;

      if (unlikely(state.single_step == state.STEP_STEPPED)) {
        state.single_step = state.STEP_NONE;
        enter_debug_mode(DCSR_CAUSE_STEP);
      }
    }
    catch (trigger_matched_t& t)
    {
      if (mmu->matched_trigger) {
        // This exception came from the MMU. That means the instruction hasn't
        // fully executed yet. We start it again, but this time it won't throw
        // an exception because matched_trigger is already set. (All memory
        // instructions are idempotent so restarting is safe.)

        insn_fetch_t fetch = mmu->load_insn(pc);
        pc = execute_insn(this, pc, fetch);
        advance_pc();

        delete mmu->matched_trigger;
        mmu->matched_trigger = NULL;
      }
      switch (state.mcontrol[t.index].action) {
        case ACTION_DEBUG_MODE:
          enter_debug_mode(DCSR_CAUSE_HWBP);
          break;
        case ACTION_DEBUG_EXCEPTION: {
          trap_breakpoint trap(state.v, t.address);
          take_trap(trap, pc);
          break;
        }
        default:
          abort();
      }
    }
    catch (wait_for_interrupt_t &t)
    {
      // Return to the outer simulation loop, which gives other devices/harts a
      // chance to generate interrupts.
      //
      // In the debug ROM this prevents us from wasting time looping, but also
      // allows us to switch to other threads only once per idle loop in case
      // there is activity.
      n = ++instret;
    }

    state.minstret->bump(instret); // state에 실행된 명령어 수 반영 
    n -= instret;
  }
}
