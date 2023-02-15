// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <arcane/ITimeLoopMng.h>
#include <arcane/IMesh.h>

class IMeshInfoPrinter
{
 public:

  virtual ~IMeshInfoPrinter() = default;
  virtual void printMeshInfo() = 0;
};

#include "MultipleMesh_axl.h"
#include "MeshInfoPrinter_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

class MultipleMeshModule
: public ArcaneMultipleMeshObject
{
 public:

  explicit MultipleMeshModule(const ModuleBuildInfo& mbi)
  : ArcaneMultipleMeshObject(mbi)
  {}

 public:

  void doLoop() override;

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultipleMeshModule::
doLoop()
{
  info() << "MultipleMeshLoop !";

  // Pour arrêter le calcul après cette itération
  subDomain()->timeLoopMng()->stopComputeLoop(true);

  options()->mesh0Printer()->printMeshInfo();
  options()->mesh1Printer()->printMeshInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MULTIPLEMESH(MultipleMeshModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DefaultMeshInfoPrinter
: public ArcaneMeshInfoPrinterObject
{
 public:

  explicit DefaultMeshInfoPrinter(const ServiceBuildInfo& sbi)
  : ArcaneMeshInfoPrinterObject(sbi)
  {}
  void printMeshInfo() override
  {
    info() << "PRINT MESH INFO!";
    IMesh* m = mesh();
    info() << "MeshName=" << m->name();
    info() << "NbCell=" << m->allCells().size();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHINFOPRINTER(DefaultMeshInfoPrinter, DefaultMeshInfoPrinter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
