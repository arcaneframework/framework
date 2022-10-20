// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.cc                                               (C) 2000-2022 */
/*                                                                           */
/* Gestion d'une commande sur accélérateur.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/RunCommand.h"

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IMemoryAllocator.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/NumArray.h"
#include "arcane/utils/ConcurrencyUtils.h"

#include "arcane/accelerator/core/RunQueueImpl.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/IReduceMemoryImpl.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/RunCommandImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file RunCommandLoop.h
 *
 * \brief Types et macros pour gérer les boucles sur les accélérateurs
 */

/*!
 * \file RunCommandEnumerate.h
 *
 * \brief Types et macros pour gérer les énumérations des entités sur les accélérateurs
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ReduceMemoryImpl
: public IReduceMemoryImpl
{
 public:

  ReduceMemoryImpl(RunCommandImpl* p)
  : m_command(p) //, m_grid_buffer(eMemoryRessource::Device)
  {
    // TODO: prendre en compte la politique d'exécution pour savoir
    // comment allouer.
    m_allocator = platform::getAcceleratorHostMemoryAllocator();
    if (!m_allocator)
      ARCANE_FATAL("No HostMemory allocator available for accelerator");
    _allocateMemoryForReduceData(128);
    m_grid_memory_info.m_grid_device_count = reinterpret_cast<unsigned int*>(m_allocator->allocate(sizeof(int)));
    (*m_grid_memory_info.m_grid_device_count) = 0;
  }
  ~ReduceMemoryImpl()
  {
    m_allocator->deallocate(m_managed_memory);
    m_allocator->deallocate(m_grid_memory_info.m_grid_device_count);
  }

 public:

  void* allocateReduceDataMemory(MemoryView identity_view) override;
  void setGridSizeAndAllocate(Int32 grid_size) override
  {
    m_grid_size = grid_size;
    _setReducePolicy();
    _allocateGridDataMemory();
  }
  Int32 gridSize() const { return m_grid_size; }

  GridMemoryInfo gridMemoryInfo() override
  {
    return m_grid_memory_info;
  }
  void release() override;

 private:

  RunCommandImpl* m_command;
  IMemoryAllocator* m_allocator;

  //! Pointeur vers la mémoire unifiée contenant la donnée réduite
  std::byte* m_managed_memory = nullptr;

  //! Taille allouée pour \a m_managed_memory
  Int64 m_size = 0;

  //! Taille courante de la grille (nombre de blocs)
  Int32 m_grid_size = 0;

  //! Taille de la donnée actuelle
  Int64 m_data_type_size = 0;

  GridMemoryInfo m_grid_memory_info;

  NumArray<Byte, MDDim1> m_grid_buffer;
  //! Buffer pour conserver la valeur de l'identité
  UniqueArray<std::byte> m_identity_buffer;

 private:

  void _allocateGridDataMemory();
  void _setReducePolicy();
  void _allocateMemoryForReduceData(Int32 new_size)
  {
    m_managed_memory = reinterpret_cast<std::byte*>(m_allocator->allocate(new_size));
    m_size = new_size;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
RunCommandImpl(RunQueueImpl* queue)
: m_queue(queue)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
~RunCommandImpl()
{
  _freePools();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_freePools()
{
  while (!m_reduce_memory_pool.empty()) {
    delete m_reduce_memory_pool.top();
    m_reduce_memory_pool.pop();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl* RunCommandImpl::
create(RunQueueImpl* r)
{
  return r->_internalCreateOrGetRunCommandImpl();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
reset()
{
  m_kernel_name = String();
  m_trace_info = TraceInfo();
  m_nb_thread_per_block = 0;
  m_parallel_loop_options = TaskFactory::defaultParallelLoopOptions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IReduceMemoryImpl* RunCommandImpl::
getOrCreateReduceMemoryImpl()
{
  ReduceMemoryImpl* p = _getOrCreateReduceMemoryImpl();
  if (p) {
    m_active_reduce_memory_list.insert(p);
  }
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
releaseReduceMemoryImpl(ReduceMemoryImpl* p)
{
  auto x = m_active_reduce_memory_list.find(p);
  if (x == m_active_reduce_memory_list.end())
    ARCANE_FATAL("ReduceMemoryImpl in not in active list");
  m_active_reduce_memory_list.erase(x);
  m_reduce_memory_pool.push(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IRunQueueStream* RunCommandImpl::
internalStream() const
{
  return m_queue->_internalStream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Runner* RunCommandImpl::
runner() const
{
  return m_queue->runner();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReduceMemoryImpl* RunCommandImpl::
_getOrCreateReduceMemoryImpl()
{
  // Pas besoin d'allouer de la mémoire spécifique si on n'est pas
  // sur un accélérateur
  if (!impl::isAcceleratorPolicy(m_queue->executionPolicy()))
    return nullptr;

  auto& pool = m_reduce_memory_pool;

  if (!pool.empty()) {
    ReduceMemoryImpl* p = pool.top();
    pool.pop();
    return p;
  }
  return new ReduceMemoryImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
release()
{
  m_command->releaseReduceMemoryImpl(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_setReducePolicy()
{
  m_grid_memory_info.m_reduce_policy = m_command->runner()->deviceReducePolicy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* ReduceMemoryImpl::
allocateReduceDataMemory(MemoryView identity_view)
{
  auto identity_span = identity_view.span();
  Int32 data_type_size = static_cast<Int32>(identity_span.size());
  m_data_type_size = data_type_size;
  if (data_type_size > m_size)
    _allocateMemoryForReduceData(data_type_size);
  // Recopie \a identity_view dans un buffer car on utilise l'asynchronisme
  // et la zone pointée par \a identity_view n'est pas forcément conservée
  m_identity_buffer.copy(identity_view.span());
  MemoryCopyArgs copy_args(m_managed_memory, m_identity_buffer.span().data(), data_type_size);
  m_command->internalStream()->copyMemory(copy_args.addAsync());
  return m_managed_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_allocateGridDataMemory()
{
  // TODO: pouvoir utiliser un padding pour éviter que les lignes de cache
  // entre les blocs se chevauchent
  Int32 total_size = CheckedConvert::toInt32(m_data_type_size * m_grid_size);
  if (total_size <= m_grid_memory_info.m_grid_memory_values.size())
    return;

  m_grid_buffer.resize(total_size);
  auto mem_view = makeMutableMemoryView(m_grid_buffer.to1DSpan());
  m_grid_memory_info.m_grid_memory_values = mem_view;
  // Indique qu'on va utiliser cette zone mémoire uniquement sur le device.
  Runner* runner = m_command->runner();
  runner->setMemoryAdvice(mem_view,eMemoryAdvice::PreferredLocationDevice);
  runner->setMemoryAdvice(mem_view,eMemoryAdvice::AccessedByHost);
  //std::cout << "RESIZE GRID t=" << total_size << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
RunCommand(RunQueue& run_queue)
: m_run_queue(run_queue)
, m_p(run_queue._getCommandImpl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand::
~RunCommand()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceInfo& RunCommand::
traceInfo() const
{
  return m_p->traceInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& RunCommand::
kernelName() const
{
  return m_p->kernelName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 RunCommand::
nbThreadPerBlock() const
{
  return m_p->m_nb_thread_per_block;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addTraceInfo(const TraceInfo& ti)
{
  m_p->m_trace_info = ti;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addKernelName(const String& v)
{
  m_p->m_kernel_name = v;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommand& RunCommand::
addNbThreadPerBlock(Int32 v)
{
  if (v<0)
    v = 0;
  if (v>0 && v<32)
    v = 32;
  m_p->m_nb_thread_per_block = v;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
setParallelLoopOptions(const ParallelLoopOptions& opt)
{
  m_p->m_parallel_loop_options = opt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ParallelLoopOptions& RunCommand::
parallelLoopOptions() const
{
  return m_p->m_parallel_loop_options;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_ACCELERATOR_CORE_EXPORT
RunCommand& operator<<(RunCommand& command,const TraceInfo& trace_info)
{
  return command.addTraceInfo(trace_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::RunCommandImpl* RunCommand::
_internalCreateImpl(impl::RunQueueImpl* queue)
{
  return new impl::RunCommandImpl(queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalDestroyImpl(impl::RunCommandImpl* p)
{
  delete p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_allocateReduceMemory(Int32 nb_grid)
{
  auto& mem_list = m_p->m_active_reduce_memory_list;
  if (!mem_list.empty()) {
    for (auto& x : mem_list)
      x->setGridSizeAndAllocate(nb_grid);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
extern "C++" IReduceMemoryImpl*
internalGetOrCreateReduceMemoryImpl(RunCommand* command)
{
  return command->m_p->getOrCreateReduceMemoryImpl();
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
