// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommand.cc                                               (C) 2000-2023 */
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
#include "arcane/utils/ForLoopTraceInfo.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryAllocator.h"

#include "arcane/accelerator/core/RunQueueImpl.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/core/IReduceMemoryImpl.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/Memory.h"
#include "arcane/accelerator/core/IRunQueueStream.h"
#include "arcane/accelerator/core/RunCommandImpl.h"
#include "arcane/accelerator/core/IRunQueueEventImpl.h"
#include "arcane/accelerator/core/internal/IRunnerRuntime.h"
#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"

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
  : m_command(p)
  , m_device_memory_bytes(eMemoryRessource::Device)
  , m_host_memory_bytes(eMemoryRessource::HostPinned)
  , m_grid_buffer(eMemoryRessource::Device)
  , m_grid_device_count(eMemoryRessource::Device)
  {
    _allocateMemoryForReduceData(128);
    _allocateMemoryForGridDeviceCount();
  }

 public:

  void* allocateReduceDataMemory(ConstMemoryView identity_view) override;
  void setGridSizeAndAllocate(Int32 grid_size) override
  {
    m_grid_size = grid_size;
    _setReducePolicy();
    _allocateGridDataMemory();
  }
  Int32 gridSize() const override { return m_grid_size; }

  GridMemoryInfo gridMemoryInfo() override
  {
    return m_grid_memory_info;
  }
  void copyReduceValueFromDevice() override;
  void release() override;

 private:

  RunCommandImpl* m_command = nullptr;

  //! Pointeur vers la mémoire unifiée contenant la donnée réduite
  std::byte* m_device_memory = nullptr;

  //! Allocation pour la donnée réduite en mémoire managée
  NumArray<std::byte, MDDim1> m_device_memory_bytes;

  //! Allocation pour la donnée réduite en mémoire hôte
  NumArray<std::byte, MDDim1> m_host_memory_bytes;

  //! Taille allouée pour \a m_device_memory
  Int64 m_size = 0;

  //! Taille courante de la grille (nombre de blocs)
  Int32 m_grid_size = 0;

  //! Taille de la donnée actuelle
  Int64 m_data_type_size = 0;

  GridMemoryInfo m_grid_memory_info;

  //! Tableau contenant la valeur de la réduction pour chaque bloc d'une grille
  NumArray<Byte, MDDim1> m_grid_buffer;

  //! Buffer pour conserver la valeur de l'identité
  UniqueArray<std::byte> m_identity_buffer;

  /*!
   * \brief Tableau de 1 entier non signé contenant le nombre de grilles ayant déja
   * effectuée la réduction.
   */
  NumArray<unsigned int, MDDim1> m_grid_device_count;

 private:

  void _allocateGridDataMemory();
  void _allocateMemoryForGridDeviceCount();
  void _setReducePolicy();
  void _allocateMemoryForReduceData(Int32 new_size)
  {
    m_device_memory_bytes.resize(new_size);
    m_device_memory = m_device_memory_bytes.to1DSpan().data();

    m_host_memory_bytes.resize(new_size);
    m_grid_memory_info.m_host_memory_for_reduced_value = m_host_memory_bytes.to1DSpan().data();

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
, m_use_accelerator(impl::isAcceleratorPolicy(queue->runner()->executionPolicy()))
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

RunCommandImpl::
~RunCommandImpl()
{
  _freePools();
  delete m_start_event;
  delete m_stop_event;
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

IRunQueueEventImpl* RunCommandImpl::
_createEvent()
{
  if (m_use_sequential_timer_event)
    return getSequentialRunQueueRuntime()->createEventImplWithTimer();
  return runner()->_createEventWithTimer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_init()
{
  // N'utilise les timers accélérateur que si le profiling est activé.
  // On fait cela pour éviter d'appeler les évènements accélérateurs car on
  // ne connait pas encore leur influence sur les performances. Si elle est
  // négligeable alors on pourra l'activer par défaut.

  // TODO: il faudrait éventuellement avoir une instance séquentielle et
  // une associée à runner() pour gérer le cas ou ProfilingRegistry::hasProfiling()
  // change en cours d'exécution.
  if (m_use_accelerator && !ProfilingRegistry::hasProfiling())
    m_use_sequential_timer_event = true;

  m_start_event = _createEvent();
  m_stop_event = _createEvent();
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
/*!
 * \brief Notification du début d'exécution de la commande.
 */
void RunCommandImpl::
notifyBeginLaunchKernel()
{
  IRunQueueStream* stream = internalStream();
  stream->notifyBeginLaunchKernel(*this);
  // TODO: utiliser la bonne stream en séquentiel
  m_start_event->recordQueue(stream);
  m_has_been_launched = true;
  if (ProfilingRegistry::hasProfiling()){
    m_begin_time = platform::getRealTimeNS();
    m_loop_one_exec_stat_ptr = &m_loop_one_exec_stat;
    m_loop_one_exec_stat.setBeginTime(m_begin_time);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification de la fin de lancement de la commande.
 *
 * La commande continue à s'exécuter en tâche de fond.
 */
void RunCommandImpl::
notifyEndLaunchKernel()
{
  IRunQueueStream* stream = internalStream();
  // TODO: utiliser la bonne stream en séquentiel
  m_stop_event->recordQueue(stream);
  stream->notifyEndLaunchKernel(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Notification de la fin d'exécution du noyau.
 *
 * Après cet appel, on est sur que la commande a fini de s'exécuter et on
 * peut la recycler. En asynchrone, cette méthode est appelée lors de la
 * synchronisation d'une file.
 */
void RunCommandImpl::
notifyEndExecuteKernel()
{  
  // Ne fait rien si la commande n'a pas été lancée.
  if (!m_has_been_launched)
    return;

  Int64 diff_time_ns = m_stop_event->elapsedTime(m_start_event);

  runner()->_addCommandTime((double)diff_time_ns / 1.0e9);

  ForLoopOneExecStat* exec_info = m_loop_one_exec_stat_ptr;
  if (exec_info){
    exec_info->setEndTime(m_begin_time+diff_time_ns);
    //std::cout << "END_EXEC exec_info=" << m_loop_run_info.traceInfo().traceInfo() << "\n";
    ForLoopTraceInfo flti(traceInfo(),kernelName());
    ProfilingRegistry::_threadLocalForLoopInstance()->merge(*exec_info,flti);
  }

  _reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommandImpl::
_reset()
{
  m_kernel_name = String();
  m_trace_info = TraceInfo();
  m_nb_thread_per_block = 0;
  m_parallel_loop_options = TaskFactory::defaultParallelLoopOptions();
  m_begin_time = 0;
  m_loop_one_exec_stat.reset();
  m_loop_one_exec_stat_ptr = nullptr;
  m_has_been_launched = false;
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
  if (!m_use_accelerator)
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
allocateReduceDataMemory(ConstMemoryView identity_view)
{
  auto identity_span = identity_view.bytes();
  Int32 data_type_size = static_cast<Int32>(identity_span.size());
  m_data_type_size = data_type_size;
  if (data_type_size > m_size)
    _allocateMemoryForReduceData(data_type_size);

  // Recopie \a identity_view dans un buffer car on utilise l'asynchronisme
  // et la zone pointée par \a identity_view n'est pas forcément conservée
  m_identity_buffer.copy(identity_view.bytes());
  MemoryCopyArgs copy_args(m_device_memory, m_identity_buffer.span().data(), data_type_size);
  m_command->internalStream()->copyMemory(copy_args.addAsync());

  return m_device_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_allocateGridDataMemory()
{
  // TODO: pouvoir utiliser un padding pour éviter que les lignes de cache
  // entre les blocs se chevauchent
  Int32 total_size = CheckedConvert::toInt32(m_data_type_size * m_grid_size);
  if (total_size <= m_grid_memory_info.m_grid_memory_values.bytes().size())
    return;

  m_grid_buffer.resize(total_size);

  auto mem_view = makeMutableMemoryView(m_grid_buffer.to1DSpan());
  m_grid_memory_info.m_grid_memory_values = mem_view;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
_allocateMemoryForGridDeviceCount()
{
  // Alloue sur le device la mémoire contenant le nombre de blocs restant à traiter
  // Il s'agit d'un seul entier non signé.
  Int64 size = sizeof(unsigned int);
  const unsigned int zero = 0;
  m_grid_device_count.resize(1);
  auto* ptr = m_grid_device_count.to1DSpan().data();
  
  m_grid_memory_info.m_grid_device_count = ptr;

  // Initialise cette zone mémoire avec 0.
  MemoryCopyArgs copy_args(ptr, &zero, size);
  m_command->internalStream()->copyMemory(copy_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReduceMemoryImpl::
copyReduceValueFromDevice()
{
  void* destination = m_grid_memory_info.m_host_memory_for_reduced_value;
  void* source = m_device_memory;
  MemoryCopyArgs copy_args(destination,source,m_data_type_size);
  m_command->internalStream()->copyMemory(copy_args);
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

void RunCommand::
_internalNotifyBeginLaunchKernel()
{
  m_p->notifyBeginLaunchKernel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void RunCommand::
_internalNotifyEndLaunchKernel()
{
  m_p->notifyEndLaunchKernel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ForLoopOneExecStat* RunCommand::
_internalCommandExecStat()
{
  return m_p->m_loop_one_exec_stat_ptr;
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
