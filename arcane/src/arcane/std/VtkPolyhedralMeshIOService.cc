// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkPolyhedralMeshIOService                      (C) 2000-2023             */
/*                                                                           */
/* Read/write fools for polyhedral mesh with vtk file format                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arccore/base/Ref.h>
#include <arccore/base/String.h>

#include "arcane/core/AbstractService.h"
#include "arcane/core/ICaseMeshReader.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IMeshBuilder.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/utils/ITraceMng.h"

namespace Arcane
{

class VtkPolyhedralMeshIOService
{
 public:

  void read(IPrimaryMesh* mesh, const String& filename) {} // todo
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VtkPolyhedralCaseMeshReader
: public AbstractService
, public ICaseMeshReader
{
 public:

  class Builder : public IMeshBuilder
  {
   public:

    explicit Builder(ITraceMng* tm, const CaseMeshReaderReadInfo& read_info)
    : m_trace_mng(tm)
    , m_read_info(read_info)
    {}

   public:

    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      build_info.addFactoryName("ArcanePolyhedralMeshFactory");
      build_info.addNeedPartitioning(false);
    }

    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      m_trace_mng->info() << "---CREATE POLYHEDRAL MESH---- " << pm->name();
      m_trace_mng->info() << "--Read mesh file " << m_read_info.fileName();
      VtkPolyhedralMeshIOService polyhedral_vtk_service{};
      polyhedral_vtk_service.read(pm, m_read_info.fileName());
    }

   private:

    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

  explicit VtkPolyhedralCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    if (read_info.format() == "vtk")
      builder = new Builder(traceMng(), read_info);
    return makeRef(builder);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(VtkPolyhedralCaseMeshReader,
                        ServiceProperty("VtkPolyhedralCaseMeshReader", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane
