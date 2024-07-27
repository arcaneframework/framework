// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Cupti.cc                                                    (C) 2000-2024 */
/*                                                                           */
/* Intégration de CUPTI.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Profiling.h"
#include "arcane/utils/internal/ProfilingInternal.h"

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#include "arcane/accelerator/core/internal/MemoryTracer.h"

#include <cuda.h>
#include <cupti.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Cuda
{
using Arcane::impl::AcceleratorStatInfoList;
namespace
{
bool global_do_print = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void arcaneCheckCudaErrors(const TraceInfo& ti, CUptiResult e)
{
  if (e == CUPTI_SUCCESS)
    return;

  const char* error_message = nullptr;
  CUptiResult e3 = cuptiGetResultString(e, &error_message);
  if (e3 != CUPTI_SUCCESS)
    error_message = "Unknown";

  ARCANE_FATAL("CUpti Error trace={0} e={1} message={2}",
               ti, e, error_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const char*
getStallReasonString(CUpti_ActivityPCSamplingStallReason reason)
{
  switch (reason) {
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID:
    return "Invalid";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
    return "Selected";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
    return "Instruction fetch";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
    return "Execution dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
    return "Memory dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
    return "Texture";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
    return "Sync";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
    return "Constant memory dependency";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
    return "Pipe busy";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
    return "Memory throttle";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
    return "Not selected";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
    return "Other";
  case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING:
    return "Sleeping";
  default:
    break;
  }

  return "<unknown>";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const char*
getUvmCounterKindString(CUpti_ActivityUnifiedMemoryCounterKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
    return "BYTES_TRANSFER_HTOD";
  case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
    return "BYTES_TRANSFER_DTOH";
  default:
    break;
  }
  return "<unknown>";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static uint64_t startTimestamp = 0;

static void
printActivity(AcceleratorStatInfoList* stat_info,
              CUpti_Activity* record, bool do_print)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
    auto* uvm = reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2*>(record);
    Int64 nb_byte = uvm->value;
    if (do_print) {
      void* address = reinterpret_cast<void*>(uvm->address);
      std::pair<String,String> mem_info = impl::MemoryTracer::findMemory(address);
      std::cout << "UNIFIED_MEMORY_COUNTER [ " << (uvm->start - startTimestamp) << " " << (uvm->end - startTimestamp) << " ]"
                << " address=" << address
                << " kind=" << getUvmCounterKindString(uvm->counterKind)
                << " value=" << nb_byte
                << " flags=" << uvm->flags
                << " source=" << uvm->srcId << " destination=" << uvm->dstId
                << " name=" << mem_info.first
                << " stack=" << mem_info.second
                << "\n";
    }
    if (stat_info) {
      if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD)
        stat_info->addMemoryTransfer(AcceleratorStatInfoList::eMemoryTransferType::HostToDevice, nb_byte);
      if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH)
        stat_info->addMemoryTransfer(AcceleratorStatInfoList::eMemoryTransferType::DeviceToHost, nb_byte);
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
    // NOTE: 'CUpti_ActivityKernel5' est disponible à partir de CUDA 11.0 mais obsolète à partir de CUDA 11.2
    // à partir de Cuda 12 on pourra utiliser 'CUpti_ActivityKernel9'.
    auto* kernel = reinterpret_cast<CUpti_ActivityKernel5*>(record);
    if (do_print) {
      std::cout << kindString << " [ " << (kernel->start - startTimestamp) << " - " << (kernel->end - startTimestamp)
                << " - " << (kernel->end - kernel->start) << " ]"
                << " device=" << kernel->deviceId << " context=" << kernel->contextId
                << " stream=" << kernel->streamId << " correlation=" << kernel->correlationId;
      std::cout << " grid=[" << kernel->gridX << "," << kernel->gridY << "," << kernel->gridZ << "]"
                << " block=[" << kernel->blockX << "," << kernel->blockY << "," << kernel->blockZ << "]"
                << " shared memory (static=" << kernel->staticSharedMemory << " dynamic=" << kernel->dynamicSharedMemory << ")"
                << " registers=" << kernel->registersPerThread
                << " name=" << '"' << kernel->name << '"'
                << "\n";
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR: {
    auto* source_locator = reinterpret_cast<CUpti_ActivitySourceLocator*>(record);
    if (do_print) {
      std::cout << "Source Locator Id " << source_locator->id
                << " File " << source_locator->fileName
                << " Line " << source_locator->lineNumber
                << "\n";
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_PC_SAMPLING: {
    auto* ps_record = reinterpret_cast<CUpti_ActivityPCSampling3*>(record);

    if (do_print) {
      std::cout << "source " << ps_record->sourceLocatorId << " functionId " << ps_record->functionId
                << " pc " << ps_record->pcOffset << " correlation " << ps_record->correlationId
                << " samples " << ps_record->samples
                << " latency samples " << ps_record->latencySamples
                << " stallreason " << getStallReasonString(ps_record->stallReason)
                << "\n";
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO: {
    auto* pcsri_result = reinterpret_cast<CUpti_ActivityPCSamplingRecordInfo*>(record);

    if (do_print) {
      std::cout << "correlation " << pcsri_result->correlationId
                << " totalSamples " << pcsri_result->totalSamples
                << " droppedSamples " << pcsri_result->droppedSamples
                << " samplingPeriodInCycles " << pcsri_result->samplingPeriodInCycles
                << "\n";
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_FUNCTION: {
    auto* func_result = reinterpret_cast<CUpti_ActivityFunction*>(record);

    if (do_print) {
      std::cout << "id " << func_result->id << " ctx " << func_result->contextId
                << " moduleId " << func_result->moduleId
                << " functionIndex " << func_result->functionIndex
                << " name " << func_result->name
                << "\n";
    }
    break;
  }

  default:
    if (do_print) {
      std::cout << "  <unknown>\n";
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe singleton pour gérer CUPTI.
 */
class CuptiInfo
{
 public:

  void init(Int32 level);
  void start();
  void stop();
  void flush();
  bool isActive() const { return m_is_active; }

 private:

  CUpti_ActivityUnifiedMemoryCounterConfig config[2];
  CUpti_ActivityPCSamplingConfig configPC;
  bool m_is_active = false;
  int m_profiling_level = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void CUPTIAPI
arcaneCuptiBufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
  const Int32 BUF_SIZE = 16 * 4096;

  // TODO: utiliser un ou plusieurs buffers pré-alloués pour éviter les
  // successions d'allocations/désallocations.
  //std::cout << "ALLOCATE BUFFER\n";
  *size = BUF_SIZE;
  *buffer = new (std::align_val_t{ 8 }) uint8_t[BUF_SIZE];
  *maxNumRecords = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static void CUPTIAPI
arcaneCuptiBufferCompleted(CUcontext ctx, uint32_t stream_id, uint8_t* buffer,
                           [[maybe_unused]] size_t size, size_t validSize)
{
  // NOTE: il semble que cette méthode soit toujours appelée depuis
  // un thread spécifique créé par le runtime CUDA.

  CUptiResult status;
  CUpti_Activity* record = nullptr;

  AcceleratorStatInfoList* stat_info = ProfilingRegistry::_threadLocalAcceleratorInstance();

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      printActivity(stat_info,record,global_do_print);
    }
    else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
    else {
      ARCANE_CHECK_CUDA(status);
    }
  } while (1);

  // report any records dropped from the queue
  size_t nb_dropped = 0;
  ARCANE_CHECK_CUDA(cuptiActivityGetNumDroppedRecords(ctx, stream_id, &nb_dropped));
  if (nb_dropped != 0)
    std::cout << "WARNING: Dropped " << nb_dropped << " activity records\n";

  delete[] buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CuptiInfo::
init(Int32 level)
{
  m_profiling_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CuptiInfo::
start()
{
  if (m_is_active)
    return;

  int level = m_profiling_level;

  ARCANE_CHECK_CUDA(cuptiActivityRegisterCallbacks(arcaneCuptiBufferRequested, arcaneCuptiBufferCompleted));

  config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
  config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
  config[0].deviceId = 0;
  config[0].enable = 1;

  config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
  config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
  config[1].deviceId = 0;
  config[1].enable = 1;

  ARCANE_CHECK_CUDA(cuptiActivityConfigureUnifiedMemoryCounter(config, 2));

  // NOTE: un seul processus peut utiliser le sampling. Si on utilise MPI avec plusieurs
  // rangs il ne faut pas activer le sampling
  if (level >= 3) {
    configPC.size = sizeof(CUpti_ActivityPCSamplingConfig);
    configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
    configPC.samplingPeriod2 = 0;
    CUcontext cuCtx;
    cuCtxGetCurrent(&cuCtx);
    ARCANE_CHECK_CUDA(cuptiActivityConfigurePCSampling(cuCtx, &configPC));
  }

  // Active les compteurs
  // CONCURRENT_KERNEL et PC_SAMPLING ne sont pas compatibles
  // Si on ajoute des compteurs ici il faut les désactiver dans stop()
  if (level >= 1)
    ARCANE_CHECK_CUDA(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
  if (level == 2)
    ARCANE_CHECK_CUDA(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  if (level >= 3)
    ARCANE_CHECK_CUDA(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

  ARCANE_CHECK_CUDA(cuptiGetTimestamp(&startTimestamp));

  // Mettre à la fin pour qu'en cas d'exception on considère l'initialisation
  // non effectuée.
  m_is_active = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CuptiInfo::
stop()
{
  if (!m_is_active)
    return;
  int level = m_profiling_level;

  if (level >= 1)
    ARCANE_CHECK_CUDA(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
  if (level == 2)
    ARCANE_CHECK_CUDA(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  if (level >= 3)
    ARCANE_CHECK_CUDA(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

  ARCANE_CHECK_CUDA(cuptiActivityFlushAll(0));
  ARCANE_CHECK_CUDA(cudaDeviceSynchronize());

  m_is_active = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CuptiInfo::
flush()
{
  // Il ne faut pas faire de flush si CUPTI n'a pas démarré car cela provoque
  // une erreur.
  if (!m_is_active)
    return;
  ARCANE_CHECK_CUDA(cuptiActivityFlushAll(0));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  CuptiInfo m_global_cupti_info;
}

extern "C++" void
initCupti(Int32 level,bool do_print)
{
  m_global_cupti_info.init(level);
  global_do_print = do_print;
}

extern "C++" void
flushCupti()
{
  m_global_cupti_info.flush();
}

extern "C++" void
startCupti()
{
  m_global_cupti_info.start();
}

extern "C++" void
stopCupti()
{
  m_global_cupti_info.stop();
}

extern "C++" bool
isCuptiActive()
{
  return m_global_cupti_info.isActive();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
