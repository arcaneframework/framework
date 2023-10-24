// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MainFactory.h                                               (C) 2000-2023 */
/*                                                                           */
/* AbstractFactory de l'implémentation par défaut d'Arcane.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_MAINFACTORY_H
#define ARCANE_IMPL_MAINFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMainFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Manufacture des classes d'Arcane.
 *
 Arcane fournit des fabriques par défaut pour la plupart des gestionnaires
 (ISuperMng, IParallelSuperMng, ...). La classe gérant le code doit par contre
 être spécifiée en implémentant la méthode createArcaneMain() dans une
 classe dérivée.

 Le point d'entrée général du code se fait par l'appel à la fonction
 arcaneMain().

 Par exemple, si on définit une classe <tt>ConcreteMainFactory</tt> qui
 dérive de MainFactory, on lance le code comme suit:
 
 * \code
 * int
 * main(int argc,char** argv)
 * {
 *   ExeInfo exe_info = ... // Création des infos de l'exécutable.
 *   ConcreteMainFactory cmf; // Création de la manufacture
 *   return Arcane::ArcaneMain::arcaneMain(exe_info,&cmf);
 * }
 * \endcode
 */
class ARCANE_IMPL_EXPORT MainFactory
: public IMainFactory
{
 public:

  MainFactory();
  ~MainFactory() override;

 public:

 public:

  //! Crée une instance de IArcaneMain
  IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) override;

 public:

  IApplication* createApplication(IArcaneMain*) override;
  IVariableMng* createVariableMng(ISubDomain*) override;
  IModuleMng* createModuleMng(ISubDomain*) override;
  IEntryPointMng* createEntryPointMng(ISubDomain*) override;
  ITimeHistoryMng* createTimeHistoryMng(ISubDomain*) override;
  ICaseMng* createCaseMng(ISubDomain*) override;
  ICaseDocument* createCaseDocument(IApplication*) override;
  ICaseDocument* createCaseDocument(IApplication*,const String& lang) override;
  ICaseDocument* createCaseDocument(IApplication*,IXmlDocumentHolder* doc) override;
  ITimeStats* createTimeStats(ISubDomain*) override;
  ITimeStats* createTimeStats(ITimerMng* tim,ITraceMng* trm,const String& name) override;
  ITimeLoopMng* createTimeLoopMng(ISubDomain*) override;
  ITimeLoop* createTimeLoop(IApplication* sm,const String& name) override;
  IIOMng* createIOMng(IApplication*) override;
  IIOMng* createIOMng(IParallelMng* pm) override;
  IServiceLoader* createServiceLoader() override;
  IServiceMng* createServiceMng(IBase*) override;
  ICheckpointMng* createCheckpointMng(ISubDomain*) override;
  IPropertyMng* createPropertyMng(ISubDomain*) override;
  Ref<IPropertyMng> createPropertyMngReference(ISubDomain*) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,IParallelMng* pm,const String& name) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name, bool is_amr) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,const String& name, eMeshAMRKind amr_type) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,IParallelMng* pm,const String& name, bool is_amr) override;
  IPrimaryMesh* createMesh(ISubDomain* sub_domain,IParallelMng* pm,const String& name, eMeshAMRKind amr_type) override;
  IMesh* createSubMesh(IMesh* mesh, const ItemGroup& group, const String& name) override;
  IDataFactory* createDataFactory(IApplication*) override;
  Ref<IDataFactoryMng> createDataFactoryMngRef(IApplication*) override;
  Ref<IAcceleratorMng> createAcceleratorMngRef(ITraceMng* tm) override;
  ITraceMng* createTraceMng() override;
  ITraceMngPolicy* createTraceMngPolicy(IApplication* app) override;
  IModuleMaster* createModuleMaster(ISubDomain* sd) override;
  ILoadBalanceMng* createLoadBalanceMng(ISubDomain* sd) override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

