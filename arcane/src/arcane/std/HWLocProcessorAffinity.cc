// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HWLocProcessorAffinity.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Gestion de l'affinité des processeurs via la bibiliothèque HWLOC.         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IProcessorAffinityService.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/AbstractService.h"

#include "arccore/base/internal/DependencyInjection.h"

#include <hwloc.h>

#include <cstdio>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HWLocProcessorAffinityService
: public TraceAccessor
, public IProcessorAffinityService
{
 public:

  explicit HWLocProcessorAffinityService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {
  }
  explicit HWLocProcessorAffinityService(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

  //<! Libère les ressources
  ~HWLocProcessorAffinityService() override
  {
    if (m_topology)
      hwloc_topology_destroy(m_topology);
  }

 public:

  void build() override {}

 public:

  void printInfos() override;
  String cpuSetString() override;
  void bindThread(Int32 cpu) override;

  Int32 numberOfCore() override;
  Int32 numberOfSocket() override;
  Int32 numberOfProcessingUnit() override;

 private:
 private:

  bool m_is_init = false;
  hwloc_topology_t m_topology = nullptr;

 private:

  void _checkInit();
  void _outputTopology(hwloc_obj_t l, hwloc_obj_t parent, int i, OStringStream& ostr);
  int _numberOf(const hwloc_obj_type_t);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Recopie temporaire de 'hwloc_obj_snprintf' qui n'existe plus
// à partir de la version 2.0 de hwloc
static int
_internal_hwloc_obj_snprintf(char* string, size_t size,
                             hwloc_obj* l, const char* _indexprefix, int verbose)
{
  const char* indexprefix = _indexprefix ? _indexprefix : "#";
  char os_index[12] = "";
  char type[64];
  char attr[128];
  int attrlen;

  if (l->os_index != (unsigned)-1) {
    std::snprintf(os_index, 12, "%s%u", indexprefix, l->os_index);
  }

  hwloc_obj_type_snprintf(type, sizeof(type), l, verbose);
  attrlen = hwloc_obj_attr_snprintf(attr, sizeof(attr), l, " ", verbose);

  if (attrlen > 0)
    return std::snprintf(string, size, "%s%s(%s)", type, os_index, attr);
  else
    return std::snprintf(string, size, "%s%s", type, os_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/* Recursively output topology in a console fashion */
void HWLocProcessorAffinityService::
_outputTopology(hwloc_obj_t l, hwloc_obj_t parent, int i, OStringStream& ostr)
{
  const char* indexprefix = "#";
  char line[512];
  if (parent && parent->arity == 1 && hwloc_bitmap_isequal(l->cpuset, parent->cpuset)) {
    ostr() << " + ";
  }
  else {
    if (parent) {
      info() << ostr.str();
      ostr.reset();
    }
    for (int z = 0; z < 2 * i; ++z)
      ostr() << ' ';
    i++;
  }
  _internal_hwloc_obj_snprintf(line, sizeof(line) - 1, l, indexprefix, 0);
  ostr() << line;
  unsigned int arity = l->arity;
  if (arity != 0 || (!i && !arity)) {
    for (unsigned int x = 0; x < arity; ++x) {
      _outputTopology(l->children[x], l, i, ostr);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HWLocProcessorAffinityService::
_checkInit()
{
  if (m_is_init)
    return;
  m_is_init = true;

  info() << "Getting hardware architecture hwloc_version=" << HWLOC_VERSION;
  int err = hwloc_topology_init(&m_topology);
  if (err != 0) {
    info() << "Error in hwloc_topology_init r=" << err;
    return;
  }
  unsigned long flags = 0;
  hwloc_topology_set_flags(m_topology, flags);

  err = hwloc_topology_load(m_topology);
  if (err != 0) {
    info() << "Error in hwloc_topology_load r=" << err;
    return;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String HWLocProcessorAffinityService::
cpuSetString()
{
  hwloc_bitmap_t new_cpuset = hwloc_bitmap_alloc();
  hwloc_get_cpubind(m_topology, new_cpuset, 0 | HWLOC_CPUBIND_THREAD);
  char result_s[1024];
  hwloc_bitmap_snprintf(result_s, sizeof(result_s) - 1, new_cpuset);
  String s{ StringView(result_s) };
  //ostr() << "R=" << result_s << '\n';
  hwloc_bitmap_free(new_cpuset);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HWLocProcessorAffinityService::
printInfos()
{
  _checkInit();
  OStringStream ostrf;
  _outputTopology(hwloc_get_root_obj(m_topology), nullptr, 0, ostrf);
  info() << ostrf.str();
  info() << "Cpuset: " << cpuSetString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HWLocProcessorAffinityService::
bindThread(Int32 cpu)
{
  int flags = 0;
  hwloc_bitmap_t new_cpuset = hwloc_bitmap_alloc();
  hwloc_get_cpubind(m_topology, new_cpuset, 0 | HWLOC_CPUBIND_THREAD);

  {
    Integer cpu_index = 0;
    unsigned int cpu_iter = 0;
    hwloc_bitmap_foreach_begin(cpu_iter, new_cpuset)
    {
      //info() << "CPUSET ITER =" << cpu_iter << " wanted=" << cpu
      //       << " is_set?=" << hwloc_cpuset_isset(new_cpuset, cpu_iter);
      if (cpu_index == cpu)
        hwloc_bitmap_only(new_cpuset, cpu_iter);
      ++cpu_index;
    }
    hwloc_bitmap_foreach_end();
  }

  hwloc_set_cpubind(m_topology, new_cpuset, flags | HWLOC_CPUBIND_THREAD);

  // Regarde si bind ok
  hwloc_get_cpubind(m_topology, new_cpuset, 0 | HWLOC_CPUBIND_THREAD);
  char result_s[1024];
  hwloc_bitmap_snprintf(result_s, sizeof(result_s) - 1, new_cpuset);
  //info() << "CPUSET V2=" << result_s;
  hwloc_bitmap_free(new_cpuset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HWLocProcessorAffinityService::
_numberOf(const hwloc_obj_type_t that)
{
  _checkInit();
  int depth = hwloc_get_type_depth(m_topology, that);
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN)
    return -1;
  int width = hwloc_get_nbobjs_by_depth(m_topology, depth);
  return width;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HWLocProcessorAffinityService::
numberOfCore(void)
{
  return _numberOf(HWLOC_OBJ_CORE);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HWLocProcessorAffinityService::
numberOfSocket()
{
  return _numberOf(HWLOC_OBJ_SOCKET);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 HWLocProcessorAffinityService::
numberOfProcessingUnit()
{
  return _numberOf(HWLOC_OBJ_PU);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_APPLICATION_FACTORY(HWLocProcessorAffinityService,
                                    IProcessorAffinityService,
                                    HWLoc);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DI_REGISTER_PROVIDER(HWLocProcessorAffinityService,
                            DependencyInjection::ProviderProperty("HWLoc"),
                            ARCANE_DI_INTERFACES(IProcessorAffinityService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
