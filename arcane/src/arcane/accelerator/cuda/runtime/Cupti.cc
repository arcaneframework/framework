// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Cupti.cc                                                    (C) 2000-2023 */
/*                                                                           */
/* Intégration de CUPTI.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#include <cuda.h>
#include <cupti.h>

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Cuda
{

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

static void
printActivity(CUpti_Activity* record)
{
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
    auto* uvm = reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2*>(record);
    std::cout << "UNIFIED_MEMORY_COUNTER [ " << (uvm->start) << " " << (uvm->end) << " ]"
              << " address=" << reinterpret_cast<void*>(uvm->address)
              << " kind=" << getUvmCounterKindString(uvm->counterKind)
              << " value=" << uvm->value
              << " source=" << uvm->srcId << " destination=" << uvm->dstId
              << "\n";
    break;
  }
  default:
    std::cout << "  <unknown>\n";
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CuptiInfo
{
 public:

  void init();

 private:

  CUpti_ActivityUnifiedMemoryCounterConfig config[2];
  bool m_is_init = false;
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
  CUptiResult status;
  CUpti_Activity* record = nullptr;

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if (status == CUPTI_SUCCESS) {
      printActivity(record);
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
init()
{
  if (m_is_init)
    return;
  m_is_init = true;

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

  // Active les compteurs
  ARCANE_CHECK_CUDA(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

  // A appeler en fin de calcul pour désactiver les compteurs
  // ARCANE_CHECK_CUDA(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  CuptiInfo m_global_cupti_info;
}

extern "C++" void
initCupti()
{
  m_global_cupti_info.init();
}
extern "C++" void
flushCupti()
{
  ARCANE_CHECK_CUDA(cuptiActivityFlushAll(0));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
