// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LinuxPerfPerformanceCounterService.cc                       (C) 2000-2022 */
/*                                                                           */
/* Récupération des compteurs hardware via l'API 'perf' de Linux.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IPerformanceCounterService.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/FactoryService.h"

#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/types.h>
#include <unistd.h>
#include <syscall.h>
#include <sys/ioctl.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LinuxPerfPerformanceCounterService
: public TraceAccessor
, public IPerformanceCounterService
{
 public:

  explicit LinuxPerfPerformanceCounterService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {
  }
  ~LinuxPerfPerformanceCounterService()
  {
    if (m_is_started)
      stop();
    _closeAll();
  }

 public:

  void build() {}

  void initialize() override
  {
    _checkInitialize();
  }

  //! Ajoute un évènement et retourne true si cela ne fonctionne pas
  bool _addEvent(int event_type, int event_config, bool is_optional = false)
  {
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    attr.type = event_type,
    attr.config = event_config;
    attr.size = sizeof(struct perf_event_attr);
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.disabled = 1;
    attr.inherit = 1;

    int cpu = -1;
    int group_fd = -1;
    unsigned long flags = 0;
    long long_fd = syscall(__NR_perf_event_open, &attr, m_process_id, cpu, group_fd, flags);
    info(4) << "AddEvent type=" << attr.type << " id=" << attr.config << " fd=" << long_fd;
    if (long_fd == (-1)){
      if (is_optional)
        return true;
      ARCANE_FATAL("ERROR for event type={0} id={1} error={2}", attr.type, attr.config, strerror(errno));
    }
    int fd = static_cast<int>(long_fd);
    m_events_file_descriptor.add(fd);
    return false;
  }

  void start() override
  {
    if (m_is_started)
      ARCANE_FATAL("start() has alredy been called");
    _checkInitialize();
    info(4) << "LinuxPerf: Start";
    for (int fd : m_events_file_descriptor) {
      int r = ::ioctl(fd, PERF_EVENT_IOC_ENABLE);
      if (r != 0)
        ARCANE_FATAL("Error starting event r={0} error={1}", r, strerror(errno));
    }
    m_is_started = true;
  }
  void stop() override
  {
    if (!m_is_started)
      ARCANE_FATAL("start() has not been called");
    info(4) << "LinuxPerf: Stop";
    for (int fd : m_events_file_descriptor) {
      int r = ::ioctl(fd, PERF_EVENT_IOC_DISABLE);
      if (r != 0)
        ARCANE_FATAL("Error stopping event r={0} error={1}", r, strerror(errno));
    }
    m_is_started = false;
  }
  bool isStarted() const override
  {
    return m_is_started;
  }

  Integer getCounters(Int64ArrayView counters, bool do_substract) override
  {
    Int32 index = 0;
    for (Int32 index = 0, n = m_events_file_descriptor.size(); index < n; ++index) {
      Int64 value = _getOneCounter(index);
      Int64 current_value = counters[index];
      counters[index] = (do_substract) ? value - current_value : value;
    }
    return index;
  }

  Int64 getCycles() override
  {
    return _getOneCounter(0);
  }

 private:

  UniqueArray<int> m_events_file_descriptor;
  int m_event_set = 0;
  bool m_is_started = false;
  bool m_is_init = false;
  pid_t m_process_id = -1;

 private:

  void _closeAll()
  {
    for (int fd : m_events_file_descriptor) {
      if (fd >= 0)
        ::close(fd);
    }
    m_events_file_descriptor.fill(-1);
  }

  Int64 _getOneCounter(Int32 index)
  {
    int fd = m_events_file_descriptor[index];
    uint64_t value[1] = { 0 };
    const size_t s = sizeof(uint64_t);
    size_t nb_read = ::read(fd, value, s);
    if (nb_read != s)
      ARCANE_FATAL("Can not read counter index={0}", index);
    return value[0];
  }

  void _checkInitialize()
  {
    if (m_is_init)
      return;

    info() << "Initialize LinuxPerfPerformanceCounterService";
    m_process_id = ::getpid();
    // Nombre de cycles CPU. Ce compteur est indispensable.
    _addEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    // Nombre d'instructions exécutées. Ce compteur est indispensable.
    _addEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);

    // Les compteurs suivants ne sont pas indispensables et il n'est
    // pas toujours facile de savoir ceux qui sont disponibles pour une
    // plateforme donnée. On essaie dans l'ordre suivant:
    // 1. Nombre de défaut du dernier niveau de cache (en général le cache L3)
    // 2. Nombre de cycles où le CPU est en attente de quelque chose.
    // 3. Nombre de défauts de cache (a priori pour tous les caches)
    const bool is_optional = true;
    bool is_bad = true;
    {
      int cache_access = (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
      is_bad = _addEvent(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | cache_access, is_optional);
    }
    if (is_bad)
      is_bad = _addEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, is_optional);
    if (is_bad)
      _addEvent(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, is_optional);

    m_is_init = true;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LinuxPerfPerformanceCounterService,
                        ServiceProperty("LinuxPerfPerformanceCounterService", ST_Application),
                        ARCANE_SERVICE_INTERFACE(IPerformanceCounterService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
