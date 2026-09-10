// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSectionService.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Service allowing the creation of a mesh with a section of another mesh.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableMetaData.h"

#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/std/MeshSection_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

  template <class T>
struct VariableGroup
{
  UniqueArray<ArrayView<T>> dim1;
  UniqueArray<Array2View<T>> dim2;
};

template <class T>
struct VariableGroupType
{
  bool isUnknownUsed() { return !(unknown.dim1.empty() && unknown.dim2.empty()); }
  bool isCellsUsed() { return !(cells.dim1.empty() && cells.dim2.empty()); }
  bool isFacesUsed() { return !(faces.dim1.empty() && faces.dim2.empty()); }
  bool isNodesUsed() { return !(nodes.dim1.empty() && nodes.dim2.empty()); }

  VariableGroup<T> unknown;
  VariableGroup<T> cells;
  VariableGroup<T> faces;
  VariableGroup<T> nodes;
};

template <class T>
struct VariableOriClone
{
  VariableGroupType<T> ori;
  VariableGroupType<T> clone;
};
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service allowing the creation of a mesh with a section of another
 * mesh.
 *
 * You can define a section of mesh with the method \a addPlan(). All cells
 * (their barycenter) between these planes will be copied in a second mesh
 * created by this service. You can get this mesh with the method
 * \a meshSection().
 */
class MeshSectionService
: public ArcaneMeshSectionObject
{
 public:

  explicit MeshSectionService(const ServiceBuildInfo& sbi)
  : ArcaneMeshSectionObject(sbi)
  , m_creation_type(sbi.creationType())
  {}

 public:

  void addPlane(const Real3& p0, const Real3& normal) override;

  void setVariables(VariableCollection variables) override;
  VariableCollection variables() override;
  void setServiceMeshUniqueId(Int32 unique_id) override;
  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }

 private:

  void _createMesh();
  void _createVariables();
  void _createCells(Int32& nb_cell, UniqueArray<Int64>& cells_infos, Int32& nb_face, UniqueArray<Int64>& faces_infos, std::unordered_map<Int64, Real3>& pos_node, UniqueArray<Cell>& ori_cells);
  void _compute();
  void _updateVariables(UniqueArray<Cell>& ori_cells);

  template <class T>
  void _updateVariablesT(UniqueArray<Cell>& ori_cells, Int32 type, T);

  template <class T>
  void _updateArrayVariable(UniqueArray<Cell>& ori_cells, T, VariableOriClone<T>& voc);

 private:

  VariableCollection m_variables_ori;
  VariableCollection m_variables_cloned;
  IPrimaryMesh* m_cloned_mesh = nullptr;
  UniqueArray<std::pair<Real3, Real3>> m_plans;
  Int32 m_mesh_uid = -1;
  eServiceType m_creation_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHSECTION(MeshSection, MeshSectionService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
addPlane(const Real3& p0, const Real3& normal)
{
  m_plans.add({ p0, math::normalizeReal3(normal) });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
setVariables(VariableCollection variables)
{
  m_variables_ori = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableCollection MeshSectionService::
variables()
{
  return m_variables_cloned;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
setServiceMeshUniqueId(Int32 unique_id)
{
  m_mesh_uid = unique_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
updateSection()
{
  _createMesh();
  _compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_createMesh()
{
  if (m_cloned_mesh != nullptr) {
    m_cloned_mesh->modifier()->clearItems();
  }

  IMeshMng* mm = subDomain()->meshMng();

  if (m_mesh_uid == -1) {
    if (m_creation_type == ST_CaseOption) {
      m_mesh_uid = options()->getUniqueIdServiceMesh();
    }
    else {
      m_mesh_uid = 0;
    }
  }

  String service_mesh_name = mesh()->name() + "_MeshSection" + m_mesh_uid;

  MeshHandle* mesh_handle = mm->findMeshHandle(service_mesh_name, false);

  if (mesh_handle == nullptr) {
    IParallelMng* pm = subDomain()->parallelMng();
    MeshBuildInfo mbi(service_mesh_name);
    mbi.addParallelMng(makeRef(pm));
    m_cloned_mesh = mm->meshFactoryMng()->createMesh(mbi);
    m_cloned_mesh->modifier()->setDynamic(true);
    m_cloned_mesh->setDimension(mesh()->dimension());
    m_cloned_mesh->endAllocate();
  }
  else {
    m_cloned_mesh = mesh_handle->mesh()->toPrimaryMesh();
    m_cloned_mesh->modifier()->clearItems();
  }
  _createVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_createVariables()
{
  IVariableMng* variable_mng = m_cloned_mesh->variableMng();
  for (VariableCollection::Enumerator i(m_variables_ori); ++i;) {
    IVariable* var = *i;
    Ref vmd(var->createMetaDataRef());
    const String& mesh_name = vmd->meshName();
    if (mesh_name.null()) {
      ARCANE_FATAL("Only variables with support are supported.");
    }
    if (vmd->isPartial()) {
      ARCANE_FATAL("Partial variables are not supported.");
    }
    const String& full_type = vmd->fullType();
    const String& base_name = vmd->baseName();
    Integer property = vmd->property();
    const String& family_name = vmd->itemFamilyName();

    // info() << "Clone variable : " << vmd->fullName();

    VariableBuildInfo vbi(m_cloned_mesh, base_name, family_name, property);
    VariableRef* variable_ref = variable_mng->_internalApi()->createVariableFromType(full_type, vbi);

    // info() << "Cloned variable : " << variable_ref->variable()->fullName();
    m_variables_cloned.add(variable_ref->variable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_createCells(Int32& sd_nb_cell, UniqueArray<Int64>& cells_infos, Int32& sd_nb_face, UniqueArray<Int64>& faces_infos, std::unordered_map<Int64, Real3>& pos_node, UniqueArray<Cell>& ori_cells)
{
  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();

  VariableFaceBool is_added(VariableBuildInfo(mesh(), "IsAdded"));
  is_added.fill(false);

  ENUMERATE_ (Cell, icell, ownCells()) {
    {
      Real3 b{ 0 };
      for (Node node : icell->nodes()) {
        b += node_coord[node];
      }
      b /= icell->nbNode();

      bool in_plan = true;
      for (auto& [p0, normal] : m_plans) {
        const Real dist = math::dot({ b - p0 }, normal);
        if (dist < 0) {
          in_plan = false;
          break;
        }
      }

      if (!in_plan)
        continue;
    }

    ori_cells.add(*icell);

    Int16 cell_type = icell->itemTypeId();
    cells_infos.add(cell_type);

    Int64 cell_uid = icell->uniqueId().asInt64();
    cells_infos.add(cell_uid);

    for (Node node : icell->nodes()) {
      Int64 node_uid = node.uniqueId().asInt64();
      cells_infos.add(node_uid);
      pos_node[node.uniqueId()] = node_coord[node];
    }
    ++sd_nb_cell;

    for (Face face : icell->faces()) {
      if (is_added[face]) continue;
      is_added[face] = true;

      Int16 face_type = face.itemTypeId();
      faces_infos.add(face_type);

      Int64 face_uid = face.uniqueId().asInt64();
      faces_infos.add(face_uid);

      for (Node node : face.nodes()) {
        Int64 node_uid = node.uniqueId().asInt64();
        faces_infos.add(node_uid);
        pos_node[node.uniqueId()] = node_coord[node];
      }
      ++sd_nb_face;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_compute()
{
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  UniqueArray<Int64> faces_infos;
  faces_infos.reserve(10000);

  UniqueArray<Cell> ori_cells;

  std::unordered_map<Int64, Real3> coord_map;

  Int32 nb_cell = 0;
  Int32 nb_face = 0;

  _createCells(nb_cell, cells_infos, nb_face, faces_infos, coord_map, ori_cells);

  m_cloned_mesh->modifier()->addFaces(nb_face, faces_infos);
  m_cloned_mesh->modifier()->addCells(nb_cell, cells_infos);
  m_cloned_mesh->modifier()->endUpdate();

  {
    VariableNodeReal3& node_coords(m_cloned_mesh->nodesCoordinates());
    ENUMERATE_ (Node, inode, m_cloned_mesh->allNodes()) {
      node_coords[inode] = coord_map[inode->uniqueId()];
    }
  }

  _updateVariables(ori_cells);

  info() << "New mesh -- NbNode : " << m_cloned_mesh->nbNode() << " -- NbCells : " << m_cloned_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_updateVariables(UniqueArray<Cell>& ori_cells)
{
  for (Int32 type = 0; type < NB_ARCANE_DATA_TYPE; ++type) {
    switch (type) {
    case DT_Byte: {
      _updateVariablesT(ori_cells, type, Byte());
    } break;
    case DT_Real: {
      _updateVariablesT(ori_cells, type, Real());
    } break;
    case DT_Real2: {
      _updateVariablesT(ori_cells, type, Real2());
    } break;
    case DT_Real2x2: {
      _updateVariablesT(ori_cells, type, Real2x2());
    } break;
    case DT_Real3: {
      _updateVariablesT(ori_cells, type, Real3());
    } break;
    case DT_Real3x3: {
      _updateVariablesT(ori_cells, type, Real3x3());
    } break;
    case DT_Int8: {
      _updateVariablesT(ori_cells, type, Int8());
    } break;
    case DT_Int16: {
      _updateVariablesT(ori_cells, type, Int16());
    } break;
    case DT_Int32: {
      _updateVariablesT(ori_cells, type, Int32());
    } break;
    case DT_Int64: {
      _updateVariablesT(ori_cells, type, Int64());
    } break;
    case DT_Float32: {
      _updateVariablesT(ori_cells, type, Float32());
    } break;
    case DT_Float16: {
      _updateVariablesT(ori_cells, type, Float16());
    } break;
    case DT_BFloat16: {
      _updateVariablesT(ori_cells, type, BFloat16());
    } break;
    default:
      break;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void MeshSectionService::
_updateVariablesT(UniqueArray<Cell>& ori_cells, Int32 type, T)
{
  VariableOriClone<T> voc;

  if (m_variables_cloned.count() != m_variables_ori.count()) {
    ARCANE_FATAL("Bad size");
  }

  VariableCollection::Enumerator iclone(m_variables_cloned);
  VariableCollection::Enumerator iori(m_variables_ori);

  while (++iclone && ++iori) {
    IVariable* ori = *iori;
    if (ori->dataType() == type) {
      IVariable* clone = *iclone;
      if (ori->dimension() == 1) {
        auto* ori_data = ARCANE_CHECK_POINTER(dynamic_cast<IArrayDataT<T>*>(ori->data()));
        auto* clo_data = ARCANE_CHECK_POINTER(dynamic_cast<IArrayDataT<T>*>(clone->data()));

        if (ori->itemKind() == IK_Unknown) {
          voc.ori.unknown.dim1.add(ori_data->view());
          voc.clone.unknown.dim1.add(clo_data->view());
        }
        else if (ori->itemKind() == IK_Cell) {
          voc.ori.cells.dim1.add(ori_data->view());
          voc.clone.cells.dim1.add(clo_data->view());
        }
        else if (ori->itemKind() == IK_Face) {
          voc.ori.faces.dim1.add(ori_data->view());
          voc.clone.faces.dim1.add(clo_data->view());
        }
        else if (ori->itemKind() == IK_Node) {
          voc.ori.nodes.dim1.add(ori_data->view());
          voc.clone.nodes.dim1.add(clo_data->view());
        }
        else {
          ARCANE_FATAL("Variable type not supported -- Type : {0}", ori->itemKind());
        }
      }
      else if (ori->dimension() == 2) {
        // if (ori->itemKind() == IK_Unknown) {
        //   VariableResizeArgs vra(ori->nbElement());
        //   clone->_internalApi()->resize(vra);
        //   auto* ori_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(ori->data()));
        //   auto* clo_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(clone->data()));
        //   voc.ori.unknown.dim2.add(ori_data->view());
        //   voc.clone.unknown.dim2.add(clo_data->view());
        // }
        // else
        if (ori->itemKind() == IK_Cell) {
          VariableResizeArgs vra(-1);
          vra.setNewSizeDim2(ori->nbElement() / mesh()->nbCell());
          clone->_internalApi()->resize(vra);
          auto* ori_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(ori->data()));
          auto* clo_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(clone->data()));
          voc.ori.cells.dim2.add(ori_data->view());
          voc.clone.cells.dim2.add(clo_data->view());
        }
        else if (ori->itemKind() == IK_Face) {
          VariableResizeArgs vra(-1);
          vra.setNewSizeDim2(ori->nbElement() / mesh()->nbFace());
          clone->_internalApi()->resize(vra);
          auto* ori_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(ori->data()));
          auto* clo_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(clone->data()));
          voc.ori.faces.dim2.add(ori_data->view());
          voc.clone.faces.dim2.add(clo_data->view());
        }
        else if (ori->itemKind() == IK_Node) {
          VariableResizeArgs vra(-1);
          vra.setNewSizeDim2(ori->nbElement() / mesh()->nbNode());
          clone->_internalApi()->resize(vra);
          auto* ori_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(ori->data()));
          auto* clo_data = ARCANE_CHECK_POINTER(dynamic_cast<IArray2DataT<T>*>(clone->data()));
          voc.ori.nodes.dim2.add(ori_data->view());
          voc.clone.nodes.dim2.add(clo_data->view());
        }
      }
      else {
        ARCANE_FATAL("Variable dim not supported -- Dim : {0}", ori->dimension());
      }
    }
  }

  _updateArrayVariable(ori_cells, T(), voc);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class T>
void MeshSectionService::
_updateArrayVariable(UniqueArray<Cell>& ori_cells, T, VariableOriClone<T>& voc)
{
  if (voc.ori.isUnknownUsed()) {
    for (Int32 i = 0; i < voc.ori.unknown.dim1.size(); ++i) {
      voc.clone.unknown.dim1[i].copy(voc.ori.unknown.dim1[i]);
    }
    for (Int32 i = 0; i < voc.ori.unknown.dim2.size(); ++i) {
      for (Int32 j = 0; j < voc.ori.unknown.dim2[i].dim2Size(); ++j) {
        voc.clone.unknown.dim2[i][j].copy(voc.ori.unknown.dim2[i][j]);
      }
    }
  }

  ENUMERATE_ (Cell, icell, m_cloned_mesh->ownCells()) {
    Cell ori_cell = ori_cells[icell.localId()];
    if (voc.ori.isCellsUsed()) {
      for (Int32 i = 0; i < voc.ori.cells.dim1.size(); ++i) {
        voc.clone.cells.dim1[i][icell.localId()] = voc.ori.cells.dim1[i][ori_cell.localId()];
      }
      for (Int32 i = 0; i < voc.ori.cells.dim2.size(); ++i) {
        voc.clone.cells.dim2[i][icell.localId()].copy(voc.ori.cells.dim2[i][ori_cell.localId()]);
      }
    }

    if (voc.ori.isNodesUsed()) {
      for (Int32 i = 0; i < icell->nbNode(); ++i) {
        Node ori_node = ori_cell.node(i);
        Node cloned_node = icell->node(i);

        for (Int32 j = 0; j < voc.ori.nodes.dim1.size(); ++j) {
          voc.clone.nodes.dim1[j][cloned_node.localId()] = voc.ori.nodes.dim1[j][ori_node.localId()];
        }
        for (Int32 j = 0; j < voc.ori.nodes.dim2.size(); ++j) {
          voc.clone.nodes.dim2[j][cloned_node.localId()].copy(voc.ori.nodes.dim2[j][ori_node.localId()]);
        }
      }
    }

    if (voc.ori.isFacesUsed()) {
      for (Int32 i = 0; i < icell->nbFace(); ++i) {
        Face ori_face = ori_cell.face(i);
        Face cloned_face = icell->face(i);

        for (Int32 j = 0; j < voc.ori.faces.dim1.size(); ++j) {
          voc.clone.faces.dim1[j][cloned_face.localId()] = voc.ori.faces.dim1[j][ori_face.localId()];
        }
        for (Int32 j = 0; j < voc.ori.faces.dim2.size(); ++j) {
          voc.clone.faces.dim2[j][cloned_face.localId()].copy(voc.ori.faces.dim2[j][ori_face.localId()]);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
