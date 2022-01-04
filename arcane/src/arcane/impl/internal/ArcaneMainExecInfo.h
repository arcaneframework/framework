// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMainExecInfo.h                                        (C) 2000-2019 */
/*                                                                           */
/* Classe gérant l'exécution.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_ARCANEMAINEXECINFO_H
#define ARCANE_IMPL_INTERNAL_ARCANEMAINEXECINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/impl/ArcaneMain.h"
#include "arcane/ApplicationBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Infos d'exécution.
 *
 * Cette classe n'est pas exportée car elle ne doit pas être utilisée
 * en dehors de cette composante.
 */
class ARCANE_IMPL_EXPORT ArcaneMainExecInfo
{
 public:
  ArcaneMainExecInfo(const ApplicationInfo& app_info,IMainFactory* factory)
  : m_app_info(app_info), m_main_factory(factory){}
  ArcaneMainExecInfo(const ApplicationInfo& app_info,const ApplicationBuildInfo& build_info,
                     IMainFactory* factory)
  : m_app_info(app_info), m_main_factory(factory),
    m_application_build_info(build_info), m_has_build_info(true){}
 public:
  int initialize();
  void execute();
  void finalize();
  int returnValue() const { return m_ret_val; }
  IArcaneMain* arcaneMainClass() const { return m_exec_main; }
  void setDirectExecFunctor(IDirectSubDomainExecuteFunctor* func) { m_direct_exec_functor = func; }
 private:
  const ApplicationInfo& m_app_info; //!< ATTENTION: référence
  IMainFactory* m_main_factory;
  ApplicationBuildInfo m_application_build_info;
  bool m_has_build_info = false;
  IArcaneMain* m_exec_main = nullptr;
  int m_ret_val = 0;
  bool m_clean_abort = false;
  IDirectSubDomainExecuteFunctor* m_direct_exec_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
