// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUtils.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Fonctions diverses sur les éléments du maillage.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/IHashAlgorithm.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/StdNum.h"
#include "arcane/core/MeshVariableInfo.h"
#include "arcane/core/Variable.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemCompare.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/ITiedInterface.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/MeshVisitor.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"
#include "arcane/core/datatype/DataAllocationInfo.h"

#include "arcane/core/VariableUtils.h"

#include <algorithm>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using impl::ItemBase;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCANE_DEBUG_MESH

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class ValueType> inline void
_writeValue(ITraceMng* trace, const String& name, ValueType v)
{
  Integer wl = CheckedConvert::toInteger(65 - name.length());
  trace->pinfo() << name << Trace::Width(wl) << v;
}

template <> inline void
_writeValue(ITraceMng* trace, const String& name, Real3 v)
{
  Integer wl = CheckedConvert::toInteger(63 - name.length());
  trace->pinfo() << name << ".x" << Trace::Width(wl) << v.x;
  trace->pinfo() << name << ".y" << Trace::Width(wl) << v.y;
  trace->pinfo() << name << ".z" << Trace::Width(wl) << v.z;
}

template <> inline void
_writeValue(ITraceMng* trace, const String& name, Real3x3 v)
{
  Integer wl = CheckedConvert::toInteger(62 - name.length());
  trace->pinfo() << name << ".xx" << Trace::Width(wl) << v.x.x;
  trace->pinfo() << name << ".xy" << Trace::Width(wl) << v.x.y;
  trace->pinfo() << name << ".xz" << Trace::Width(wl) << v.x.z;

  trace->pinfo() << name << ".yx" << Trace::Width(wl) << v.y.x;
  trace->pinfo() << name << ".yy" << Trace::Width(wl) << v.y.y;
  trace->pinfo() << name << ".yz" << Trace::Width(wl) << v.y.z;

  trace->pinfo() << name << ".zx" << Trace::Width(wl) << v.z.x;
  trace->pinfo() << name << ".zy" << Trace::Width(wl) << v.z.y;
  trace->pinfo() << name << ".zz" << Trace::Width(wl) << v.z.z;
}

template <class ItemType, class ValueType> inline void
_writeInfo(ISubDomain* mng, const VariableCollection& variables, const ItemType& item)
{
  typedef typename MeshVariableInfoT<ItemType, ValueType, 0>::PrivateType VariableScalarTrueType;
  typedef typename MeshVariableInfoT<ItemType, ValueType, 1>::PrivateType VariableArrayTrueType;

  ITraceMng* trace = mng->traceMng();
  eItemKind item_kind = ItemTraitsT<ItemType>::kind();

  for (VariableCollection::Enumerator i(variables); ++i;) {
    IVariable* vp = *i;
    // Pour l'instant, n'affiche pas les variables partielles
    if (vp->isPartial())
      continue;
    // Vérifie qu'il s'agit bien d'une variable du maillage et du bon genre
    ItemGroup group = vp->itemGroup();
    if (group.itemKind() != item_kind)
      continue;

    const String& name = vp->name();
    auto* v = dynamic_cast<VariableScalarTrueType*>(vp);
    if (v) {
      ConstArrayView<ValueType> values = v->valueView();
      if (values.size() < item.localId()) {
        trace->error() << "Invalid dimensions for variable '" << name << "' "
                       << "(size=" << values.size() << " index=" << item.localId();
        continue;
      }
      _writeValue(trace, name, values[item.localId()]);
    }

    auto* v2 = dynamic_cast<VariableArrayTrueType*>(vp);
    if (v2) {
      Array2View<ValueType> values = v2->valueView();
      Integer lid = item.localId();
      if (values.dim1Size() < lid) {
        trace->error() << "Invalid dimensions for variable '" << name << "' "
                       << "(size=" << values.dim1Size() << " index=" << lid;
        continue;
      }
      Integer n = values[lid].size();
      for (Integer z = 0; z < n; ++z) {
        _writeValue(trace, name + "[" + z + "]", values[lid][z]);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshItemInfo(ISubDomain* sd, Cell cell, bool depend_info)
{
  ITraceMng* trace = sd->traceMng();
  Integer sid = sd->subDomainId();
  StringBuilder buf("Info-");
  buf += sid;
  Trace::Setter mci(trace, buf.toString());

  VariableCollection variables(sd->variableMng()->usedVariables());

  Integer nb_node = cell.nbNode();
  Integer nb_edge = cell.nbEdge();
  Integer nb_face = cell.nbFace();

  trace->pinfo() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  trace->pinfo() << "** -- Dumping information for cell " << cell.uniqueId();
  trace->pinfo() << "** -- Structure:";
  trace->pinfo() << "unique id:               " << Trace::Width(5) << cell.uniqueId();
  trace->pinfo() << "local id:                " << Trace::Width(5) << cell.localId();
  trace->pinfo() << "owner:                   " << Trace::Width(5) << cell.owner();
  trace->pinfo() << "type:                    " << Trace::Width(5) << cell.typeInfo()->typeName();

  trace->pinfo() << "number of nodes:         " << Trace::Width(5) << nb_node;
  for (Integer i = 0; i < nb_node; ++i)
    trace->pinfo() << "unique id of node " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << cell.node(i).uniqueId();

  trace->pinfo() << "number of edges:         " << Trace::Width(5) << nb_edge;
  for (Integer i = 0; i < nb_edge; ++i)
    trace->pinfo() << "unique id of edge " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << cell.edge(i).uniqueId();

  trace->pinfo() << "number of faces:         " << Trace::Width(5) << nb_face;
  for (Integer i = 0; i < nb_face; ++i)
    trace->pinfo() << "local id of face " << Trace::Width(2)
                   << i << " :    " << Trace::Width(5) << cell.face(i).localId();

  trace->pinfo() << "** -- Variables:";
  _writeInfo<Cell, Real>(sd, variables, cell);
  _writeInfo<Cell, Real2>(sd, variables, cell);
  _writeInfo<Cell, Real3>(sd, variables, cell);
  _writeInfo<Cell, Real2x2>(sd, variables, cell);
  _writeInfo<Cell, Real3x3>(sd, variables, cell);
  _writeInfo<Cell, Int32>(sd, variables, cell);
  _writeInfo<Cell, Int64>(sd, variables, cell);

  trace->pinfo() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";

  if (depend_info) {
    for (Integer i = 0; i < nb_node; ++i)
      writeMeshItemInfo(sd, cell.node(i), false);
    for (Integer i = 0; i < nb_edge; ++i)
      writeMeshItemInfo(sd, cell.edge(i), false);
    for (Integer i = 0; i < nb_face; ++i)
      writeMeshItemInfo(sd, cell.face(i), false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshItemInfo(ISubDomain* sd, Node node, bool depend_info)
{
  ITraceMng* trace = sd->traceMng();
  Integer sid = sd->subDomainId();
  StringBuilder buf("Info-");
  buf += sid;
  Trace::Setter mci(trace, buf.toString());

  VariableCollection variables(sd->variableMng()->usedVariables());

  Integer nb_cell = node.nbCell();
  Integer nb_edge = node.nbEdge();
  Integer nb_face = node.nbFace();

  trace->pinfo() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  trace->pinfo() << "** -- Dumping information for node " << node.uniqueId();
  trace->pinfo() << "** -- Structure:";
  trace->pinfo() << "unique id:               " << Trace::Width(5) << node.uniqueId();
  trace->pinfo() << "local id:                " << Trace::Width(5) << node.localId();
  trace->pinfo() << "owner:                   " << Trace::Width(5) << node.owner();

  trace->pinfo() << "number of cells:         " << Trace::Width(5) << nb_cell;
  for (Integer i = 0; i < nb_cell; ++i)
    trace->pinfo() << "unique id of cell " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << node.cell(i).uniqueId();

  trace->pinfo() << "number of faces:         " << Trace::Width(5) << nb_face;
  for (Integer i = 0; i < nb_face; ++i)
    trace->pinfo() << "local id of face " << Trace::Width(2)
                   << i << " :    " << Trace::Width(5) << node.face(i).localId();

  trace->pinfo() << "** -- Variables:";
  _writeInfo<Node, Real>(sd, variables, node);
  _writeInfo<Node, Real2>(sd, variables, node);
  _writeInfo<Node, Real3>(sd, variables, node);
  _writeInfo<Node, Real2x2>(sd, variables, node);
  _writeInfo<Node, Real3x3>(sd, variables, node);
  _writeInfo<Node, Int32>(sd, variables, node);
  _writeInfo<Node, Int64>(sd, variables, node);

  if (depend_info) {
    for (Integer i = 0; i < nb_cell; ++i)
      writeMeshItemInfo(sd, node.cell(i), false);
    for (Integer i = 0; i < nb_edge; ++i)
      writeMeshItemInfo(sd, node.edge(i), false);
    for (Integer i = 0; i < nb_face; ++i)
      writeMeshItemInfo(sd, node.face(i), false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshItemInfo(ISubDomain* sd, Edge edge, bool depend_info)
{
  ITraceMng* trace = sd->traceMng();
  Integer sid = sd->subDomainId();
  StringBuilder buf("Info-");
  buf += sid;
  Trace::Setter mci(trace, buf.toString());

  VariableCollection variables(sd->variableMng()->usedVariables());

  Integer nb_cell = edge.nbCell();
  Integer nb_face = edge.nbFace();
  Integer nb_node = edge.nbNode();

  trace->pinfo() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  trace->pinfo() << "** -- Dumping information for face " << edge.localId() << " (localid)";
  trace->pinfo() << "** -- Structure:";
  trace->pinfo() << "unique id:               " << Trace::Width(5) << "None";
  trace->pinfo() << "local id:                " << Trace::Width(5) << edge.localId();
  trace->pinfo() << "owner:                   " << Trace::Width(5) << edge.owner();

  trace->pinfo() << "number of nodes:         " << Trace::Width(5) << nb_node;
  for (Integer i = 0; i < nb_node; ++i)
    trace->pinfo() << "unique id of node " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << edge.node(i).uniqueId();

  trace->pinfo() << "number of cells:         " << Trace::Width(5) << nb_cell;
  for (Integer i = 0; i < nb_cell; ++i)
    trace->pinfo() << "unique id of cell " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << edge.cell(i).uniqueId();

  trace->pinfo() << "** -- Variables:";
  _writeInfo<Edge, Real>(sd, variables, edge);
  _writeInfo<Edge, Real2>(sd, variables, edge);
  _writeInfo<Edge, Real3>(sd, variables, edge);
  _writeInfo<Edge, Real2x2>(sd, variables, edge);
  _writeInfo<Edge, Real3x3>(sd, variables, edge);
  _writeInfo<Edge, Int32>(sd, variables, edge);
  _writeInfo<Edge, Int64>(sd, variables, edge);

  if (depend_info) {
    for (Integer i = 0; i < nb_node; ++i)
      writeMeshItemInfo(sd, edge.node(i), false);
    for (Integer i = 0; i < nb_face; ++i)
      writeMeshItemInfo(sd, edge.face(i), false);
    for (Integer i = 0; i < nb_cell; ++i)
      writeMeshItemInfo(sd, edge.cell(i), false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshItemInfo(ISubDomain* sd, Face face, bool depend_info)
{
  ITraceMng* trace = sd->traceMng();
  Integer sid = sd->subDomainId();
  StringBuilder buf("Info-");
  buf += sid;
  Trace::Setter mci(trace, buf.toString());

  VariableCollection variables(sd->variableMng()->usedVariables());

  Integer nb_cell = face.nbCell();
  Integer nb_edge = face.nbEdge();
  Integer nb_node = face.nbNode();

  trace->pinfo() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  trace->pinfo() << "** -- Dumping information for face " << face.localId() << " (localid)";
  trace->pinfo() << "** -- Structure:";
  trace->pinfo() << "unique id:               " << Trace::Width(5) << "None";
  trace->pinfo() << "local id:                " << Trace::Width(5) << face.localId();
  trace->pinfo() << "owner:                   " << Trace::Width(5) << face.owner();

  trace->pinfo() << "number of nodes:         " << Trace::Width(5) << nb_node;
  for (Integer i = 0; i < nb_node; ++i)
    trace->pinfo() << "unique id of node " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << face.node(i).uniqueId();

  trace->pinfo() << "number of cells:         " << Trace::Width(5) << nb_cell;
  for (Integer i = 0; i < nb_cell; ++i)
    trace->pinfo() << "unique id of cell " << Trace::Width(2)
                   << i << " :   " << Trace::Width(5) << face.cell(i).uniqueId();

  trace->pinfo() << "** -- Variables:";
  _writeInfo<Face, Real>(sd, variables, face);
  _writeInfo<Face, Real2>(sd, variables, face);
  _writeInfo<Face, Real3>(sd, variables, face);
  _writeInfo<Face, Real2x2>(sd, variables, face);
  _writeInfo<Face, Real3x3>(sd, variables, face);
  _writeInfo<Face, Int32>(sd, variables, face);
  _writeInfo<Face, Int64>(sd, variables, face);

  if (depend_info) {
    for (Integer i = 0; i < nb_node; ++i)
      writeMeshItemInfo(sd, face.node(i), false);
    for (Integer i = 0; i < nb_edge; ++i)
      writeMeshItemInfo(sd, face.edge(i), false);
    for (Integer i = 0; i < nb_cell; ++i)
      writeMeshItemInfo(sd, face.cell(i), false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class _CompareNodes
{
 public:

  bool operator()(const Node& node1, const Node& node2)
  {
    const Real3& r1 = m_coords[node1];
    const Real3& r2 = m_coords[node2];
    return r1 < r2;
  }

 public:

  _CompareNodes(IMesh* mesh, const SharedVariableNodeReal3& coords)
  : m_items(mesh->itemsInternal(IK_Node))
  , m_coords(coords)
  {
  }
  ItemInternalList m_items;
  const SharedVariableNodeReal3& m_coords;
};

class _CompareItemWithNodes
{
 public:

  bool operator()(const ItemWithNodes& cell1, const ItemWithNodes& cell2)
  {
    Integer nb1 = cell1.nbNode();
    Integer nb2 = cell2.nbNode();
    if (nb1 < nb2)
      return true;
    if (nb2 < nb1)
      return false;
    Integer n = nb1;
    for (Integer i = 0; i < n; ++i) {
      Integer n1 = m_nodes_sorted_id[cell1.node(i).localId()];
      Integer n2 = m_nodes_sorted_id[cell2.node(i).localId()];
      if (n1 != n2)
        return n1 < n2;
    }
    return false;
    //return n1<n2;
  }

 public:

  _CompareItemWithNodes(IMesh* mesh, eItemKind ik, Int32ConstArrayView nodes_sorted_id)
  : m_items(mesh->itemsInternal(ik))
  , m_nodes_sorted_id(nodes_sorted_id)
  {
  }
  ItemInternalList m_items;
  Int32ConstArrayView m_nodes_sorted_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void _writeItems(std::ostream& ofile, const String& name, Int32ConstArrayView ids)
{
  Integer nb_sub_item = ids.size();
  ofile << " " << name << " " << nb_sub_item << " (";
  for (Integer i = 0; i < nb_sub_item; ++i)
    ofile << ' ' << ids[i];
  ofile << " )";
}

class IItemFiller
{
 public:

  virtual ~IItemFiller() {}

 public:

  virtual Integer nbItem(ItemBase item) const = 0;
  virtual void fillLocalIds(ItemBase item, Int32ArrayView ids) const = 0;
};

class EdgeFiller
: public IItemFiller
{
  Integer nbItem(ItemBase item) const override { return item.nbEdge(); }
  void fillLocalIds(ItemBase item, Int32ArrayView ids) const override
  {
    Integer nb_edge = ids.size();
    for (Integer i = 0; i < nb_edge; ++i)
      ids[i] = item.edgeId(i);
  }
};

class CellFiller
: public IItemFiller
{
  Int32 nbItem(ItemBase item) const override { return item.nbCell(); }
  void fillLocalIds(ItemBase item, Int32ArrayView ids) const override
  {
    Integer nb_cell = ids.size();
    for (Integer i = 0; i < nb_cell; ++i)
      ids[i] = item.cellId(i);
  }
};

class FaceFiller
: public IItemFiller
{
  Int32 nbItem(ItemBase item) const override { return item.nbFace(); }
  void fillLocalIds(ItemBase item, Int32ArrayView ids) const override
  {
    Integer nb_face = ids.size();
    for (Integer i = 0; i < nb_face; ++i)
      ids[i] = item.faceId(i);
  }
};

class NodeFiller
: public IItemFiller
{
  virtual Int32 nbItem(ItemBase item) const { return item.nbNode(); }
  virtual void fillLocalIds(ItemBase item, Int32ArrayView ids) const
  {
    Integer nb_node = ids.size();
    for (Integer i = 0; i < nb_node; ++i)
      ids[i] = item.nodeId(i);
  }
};

void _fillSorted(Item titem, Int32Array& local_ids, Int32ConstArrayView sorted_ids, const IItemFiller& filler)
{
  ItemBase item = titem.itemBase();
  Integer n = filler.nbItem(item);
  local_ids.resize(n);
  filler.fillLocalIds(item, local_ids);
  for (Integer i = 0; i < n; ++i)
    local_ids[i] = sorted_ids[local_ids[i]];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshInfosSorted(IMesh* mesh, const String& file_name)
{
  if (!mesh)
    return;
  std::ofstream ofile(file_name.localstr());
  ofile.precision(FloatInfo<Real>::maxDigit() + 1);
  Integer nb_node = mesh->nbNode();
  Integer nb_edge = mesh->nbEdge();
  Integer nb_face = mesh->nbFace();
  Integer nb_cell = mesh->nbCell();
  ofile << "** Mesh Sorted --> "
        << " Nodes " << nb_node
        << " Edges " << nb_edge
        << " Faces " << nb_face
        << " Cells " << nb_cell
        << "\n";
  UniqueArray<Node> sorted_nodes(nb_node);
  UniqueArray<Edge> sorted_edges(nb_edge);
  UniqueArray<Face> sorted_faces(nb_face);
  UniqueArray<Cell> sorted_cells(nb_cell);

  Int32UniqueArray nodes_sorted_id(nb_node);
  Int32UniqueArray edges_sorted_id(nb_edge);
  Int32UniqueArray faces_sorted_id(nb_face);
  Int32UniqueArray cells_sorted_id(nb_cell);
  {
    // Trie les noeuds par ordre croissant de leur coordonnées
    NodeInfoListView nodes(mesh->nodeFamily());
    for (Integer i = 0; i < nb_node; ++i)
      sorted_nodes[i] = nodes[i];
    {
      SharedVariableNodeReal3 shared_coords(mesh->sharedNodesCoordinates());
      _CompareNodes compare_nodes(mesh, shared_coords);
      std::sort(std::begin(sorted_nodes), std::end(sorted_nodes), compare_nodes);
    }
    for (Integer i = 0; i < nb_node; ++i)
      nodes_sorted_id[sorted_nodes[i].localId()] = i;

    // Trie les arêtes
    EdgeInfoListView edges(mesh->edgeFamily());
    for (Integer i = 0; i < nb_edge; ++i)
      sorted_edges[i] = edges[i];
    {
      _CompareItemWithNodes compare_edges(mesh, IK_Edge, nodes_sorted_id);
      std::sort(std::begin(sorted_edges), std::end(sorted_edges), compare_edges);
    }
    for (Integer i = 0; i < nb_edge; ++i)
      edges_sorted_id[sorted_edges[i].localId()] = i;

    // Trie les faces
    FaceInfoListView faces(mesh->faceFamily());
    for (Integer i = 0; i < nb_face; ++i)
      sorted_faces[i] = faces[i];
    {
      _CompareItemWithNodes compare_faces(mesh, IK_Face, nodes_sorted_id);
      std::sort(std::begin(sorted_faces), std::end(sorted_faces), compare_faces);
    }
    for (Integer i = 0; i < nb_face; ++i)
      faces_sorted_id[sorted_faces[i].localId()] = i;

    // Trie les mailles
    CellInfoListView cells(mesh->cellFamily());
    for (Integer i = 0; i < nb_cell; ++i)
      sorted_cells[i] = cells[i];
    {
      _CompareItemWithNodes compare_cells(mesh, IK_Cell, nodes_sorted_id);
      std::sort(std::begin(sorted_cells), std::end(sorted_cells), compare_cells);
    }
    for (Integer i = 0; i < nb_cell; ++i)
      cells_sorted_id[sorted_cells[i].localId()] = i;

    SharedVariableNodeReal3 coords(mesh->sharedNodesCoordinates());

    ofile << "** Nodes\n";
    Int32UniqueArray lids;
    for (Integer i = 0; i < nb_node; ++i) {
      const Node& node = sorted_nodes[i];
      ofile << "Node: " << i << " Coord: " << coords[node];

      _fillSorted(node, lids, edges_sorted_id, EdgeFiller());
      std::sort(std::begin(lids), std::end(lids));
      _writeItems(ofile, "Edges", lids);

      _fillSorted(node, lids, faces_sorted_id, FaceFiller());
      std::sort(std::begin(lids), std::end(lids));
      _writeItems(ofile, "Faces", lids);

      _fillSorted(node, lids, cells_sorted_id, CellFiller());
      std::sort(std::begin(lids), std::end(lids));
      _writeItems(ofile, "Cells", lids);

      ofile << '\n';
    }

    ofile << "** Edges\n";
    for (Integer i = 0; i < nb_edge; ++i) {
      Edge edge = sorted_edges[i];
      Integer edge_nb_node = edge.nbNode();
      Integer edge_nb_face = edge.nbFace();
      Integer edge_nb_cell = edge.nbCell();
      ofile << "Edge: " << i
            << " Nodes " << edge_nb_node << " (";
      for (Integer i_node = 0; i_node < edge_nb_node; ++i_node)
        ofile << ' ' << nodes_sorted_id[edge.node(i_node).localId()];
      ofile << " )";
      ofile << " Faces " << edge_nb_face << " (";
      for (Integer i_face = 0; i_face < edge_nb_face; ++i_face)
        ofile << ' ' << faces_sorted_id[edge.face(i_face).localId()];
      ofile << " )";
      ofile << " Cells " << edge_nb_cell << " (";
      for (Integer i_cell = 0; i_cell < edge_nb_cell; ++i_cell)
        ofile << ' ' << cells_sorted_id[edge.cell(i_cell).localId()];
      ofile << " )";
      ofile << '\n';
    }

    ofile << "** Faces\n";
    for (Integer i = 0; i < nb_face; ++i) {
      Face face = sorted_faces[i];
      Integer face_nb_node = face.nbNode();
      Integer face_nb_edge = face.nbEdge();
      Integer face_nb_cell = face.nbCell();
      ofile << "Face: " << i;
      ofile << " Nodes " << face.nbNode() << " (";
      for (Integer i_node = 0; i_node < face_nb_node; ++i_node)
        ofile << ' ' << nodes_sorted_id[face.node(i_node).localId()];
      ofile << " )";

      ofile << " Edges " << face_nb_edge << " (";
      for (Integer i_edge = 0; i_edge < face_nb_edge; ++i_edge)
        ofile << ' ' << edges_sorted_id[face.edge(i_edge).localId()];
      ofile << " )";

      ofile << " Cells " << face_nb_cell << " (";
      for (Integer i_cell = 0; i_cell < face_nb_cell; ++i_cell)
        ofile << ' ' << cells_sorted_id[face.cell(i_cell).localId()];

      const Cell& back_cell = face.backCell();
      if (!back_cell.null())
        ofile << " Back " << cells_sorted_id[back_cell.localId()];

      const Cell& front_cell = face.frontCell();
      if (!front_cell.null())
        ofile << " Front " << cells_sorted_id[front_cell.localId()];

      ofile << " )";
      ofile << '\n';
    }

    ofile << "** Cells\n";
    for (Integer i = 0; i < nb_cell; ++i) {
      Cell cell = sorted_cells[i];
      //Integer cell_nb_node = cell.nbNode();
      ofile << "Cell: " << i;

      _fillSorted(cell, lids, nodes_sorted_id, NodeFiller());
      _writeItems(ofile, "Nodes", lids);
      _fillSorted(cell, lids, edges_sorted_id, EdgeFiller());
      _writeItems(ofile, "Edges", lids);
      _fillSorted(cell, lids, faces_sorted_id, FaceFiller());
      _writeItems(ofile, "Faces", lids);
      ofile << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshInfos(IMesh* mesh, const String& file_name)
{
  if (!mesh)
    return;
  std::ofstream ofile(file_name.localstr());
  ofile.precision(FloatInfo<Real>::maxDigit() + 1);
  Integer nb_node = mesh->nbNode();
  Integer nb_edge = mesh->nbEdge();
  Integer nb_face = mesh->nbFace();
  Integer nb_cell = mesh->nbCell();
  ofile << "** Mesh --> "
        << " Nodes " << nb_node
        << " Edges " << nb_edge
        << " Faces " << nb_face
        << " Cells " << nb_cell
        << "\n";

  NodeInfoListView nodes(mesh->nodeFamily());
  FaceInfoListView faces(mesh->faceFamily());
  CellInfoListView cells(mesh->cellFamily());
  //TODO pouvoir afficher les infos même si on n'est pas un maillage primaire
  VariableNodeReal3& coords(mesh->toPrimaryMesh()->nodesCoordinates());

  ofile << "** Nodes\n";
  for (Integer i = 0; i < nb_node; ++i) {
    Node node = nodes[i];
    Integer node_nb_face = node.nbFace();
    Integer node_nb_cell = node.nbCell();
    ofile << "Node: " << i << " Coord: " << coords[node];
    ofile << " Faces " << node_nb_face << " (";
    for (Integer i_face = 0; i_face < node_nb_face; ++i_face)
      ofile << ' ' << node.face(i_face).localId();
    ofile << " )";
    ofile << " Cells " << node_nb_cell << " (";
    for (Integer i_cell = 0; i_cell < node_nb_cell; ++i_cell)
      ofile << ' ' << node.cell(i_cell).localId();
    ofile << " )";
    ofile << '\n';
  }

  ofile << "** Faces\n";
  for (Integer i = 0; i < nb_face; ++i) {
    Face face = faces[i];
    Integer face_nb_node = face.nbNode();
    Integer face_nb_cell = face.nbCell();
    ofile << "Face: " << i
          << " Nodes " << face_nb_node << " (";
    for (Integer i_node = 0; i_node < face_nb_node; ++i_node)
      ofile << ' ' << face.node(i_node).localId();
    ofile << " )";
    ofile << " Cells " << face_nb_cell << " (";

    for (Integer i_cell = 0; i_cell < face_nb_cell; ++i_cell)
      ofile << ' ' << face.cell(i_cell).localId();

    const Cell& back_cell = face.backCell();
    if (!back_cell.null())
      ofile << " Back " << back_cell.localId();

    const Cell& front_cell = face.frontCell();
    if (!front_cell.null())
      ofile << " Front " << front_cell.localId();

    ofile << " )";
    ofile << '\n';
  }

  ofile << "** Cells\n";
  for (Integer i = 0; i < nb_cell; ++i) {
    Cell cell = cells[i];
    Integer cell_nb_node = cell.nbNode();
    Integer cell_nb_face = cell.nbFace();
    ofile << "Cell: " << i
          << " Nodes " << cell_nb_node << " (";
    for (Integer i_node = 0; i_node < cell_nb_node; ++i_node)
      ofile << ' ' << cell.node(i_node).localId();
    ofile << " )";
    ofile << " Faces " << cell_nb_face << " (";
    for (Integer i_face = 0; i_face < cell_nb_face; ++i_face)
      ofile << ' ' << cell.face(i_face).localId();
    ofile << " )";
    ofile << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  template <typename ItemType> void
  _sortByUniqueIds(IMesh* mesh, eItemKind ik, Array<ItemType>& items)
  {
    ItemGroup all_items(mesh->itemFamily(ik)->allItems());
    items.resize(all_items.size());

    Integer index = 0;
    ENUMERATE_ (ItemType, i, all_items) {
      ItemType item = *i;
      items[index] = item;
      ++index;
    }
    std::sort(std::begin(items), std::end(items), ItemCompare());
  }

  void
  _stringToIds(const String& str, Int64Array& ids)
  {
    ids.clear();
    std::istringstream istr(str.localstr());
    Integer z = 0;
    while (istr.good()) {
      istr >> z;
      if (!istr)
        break;
      ids.add(z);
    }
  }

  template <typename SubItemType> void
  _writeSubItems(std::ostream& ofile, const char* item_name, ItemConnectedListViewT<SubItemType> sub_list)
  {
    Int32 n = sub_list.size();
    if (n == 0)
      return;
    ofile << "<" << item_name << " count='" << n << "'>";
    for (SubItemType sub_item : sub_list)
      ofile << ' ' << sub_item.uniqueId();
    ofile << "</" << item_name << ">";
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
writeMeshConnectivity(IMesh* mesh, const String& file_name)
{
  std::ofstream ofile(file_name.localstr());
  ofile.precision(FloatInfo<Real>::maxDigit() + 1);
  if (!mesh)
    return;

  ITraceMng* trace = mesh->traceMng();

  trace->info() << "Writing mesh connectivity in '" << file_name << "'";

  ofile << "<?xml version='1.0' ?>\n";
  ofile << "<mesh-connectivity>\n";
  UniqueArray<Node> nodes;
  UniqueArray<Edge> edges;
  UniqueArray<Face> faces;
  UniqueArray<Cell> cells;

  _sortByUniqueIds(mesh, IK_Node, nodes);
  _sortByUniqueIds(mesh, IK_Edge, edges);
  _sortByUniqueIds(mesh, IK_Face, faces);
  _sortByUniqueIds(mesh, IK_Cell, cells);

  // Écrit les noeuds
  {
    ofile << "<nodes count='" << nodes.size() << "'>\n";
    for (Node item : nodes) {
      ofile << " <node uid='" << item.uniqueId() << "' owner='" << item.owner() << "'>";
      _writeSubItems(ofile, "cells", item.cells());
      _writeSubItems(ofile, "faces", item.faces());
      _writeSubItems(ofile, "edges", item.edges());
      ofile << "</node>\n";
    }
    ofile << "</nodes>\n";
  }

  // Écrit les arêtes
  {
    ofile << "<edges count='" << edges.size() << "'>\n";
    for (Edge edge : edges) {
      ofile << " <edge uid='" << edge.uniqueId() << "' owner='" << edge.owner() << "'>";
      _writeSubItems(ofile, "nodes", edge.nodes());
      _writeSubItems(ofile, "cells", edge.cells());
      _writeSubItems(ofile, "faces", edge.faces());
      ofile << "</edge>\n";
    }
    ofile << "</edges>\n";
  }

  // Écrit les faces
  {
    ofile << "<faces count='" << faces.size() << "'>\n";
    for (Face face : faces) {
      //      Integer item_nb_face = item.nbFace();
      ofile << " <face uid='" << face.uniqueId()
            << "' typeid='" << face.type()
            << "' owner='" << face.owner() << "'>";
      _writeSubItems(ofile, "nodes", face.nodes());
      _writeSubItems(ofile, "edges", face.edges());
      {
        // Infos sur les mailles
        ofile << "<cells";
        Cell back_cell = face.backCell();
        if (!back_cell.null())
          ofile << " back='" << back_cell.uniqueId() << "'";
        Cell front_cell = face.frontCell();
        if (!front_cell.null())
          ofile << " front='" << front_cell.uniqueId() << "'";
        ofile << "/>";
      }

      // Infos sur les maitres/esclaves
      if (face.isSlaveFace())
        _writeSubItems(ofile, "slave-faces", face.slaveFaces());
      if (face.isMasterFace()) {
        ofile << "<faces count='1'>";
        ofile << ' ' << face.masterFace().uniqueId();
        ofile << "</faces>";
      }

      ofile << "</face>\n";
    }
    ofile << "</faces>\n";
  }

  // Écrit les mailles
  {
    ofile << "<cells count='" << cells.size() << "'>\n";
    // Pour les mailles autour d'une maille.
    // Une maille est autour d'une autre, si elle est connectée par
    // au moins un noeud
    Int64UniqueArray ghost_cells_layer1;
    ghost_cells_layer1.reserve(100);
    for (Cell cell : cells) {
      ofile << " <cell uid='" << cell.uniqueId()
            << "' typeid='" << cell.type()
            << "' owner='" << cell.owner() << "'>";
      _writeSubItems(ofile, "nodes", cell.nodes());
      _writeSubItems(ofile, "edges", cell.edges());
      _writeSubItems(ofile, "faces", cell.faces());
      if (mesh->isAmrActivated()) {
        ofile << "<amr level='" << cell.level() << "'>";
        // {
        //   ofile << "<parent count='" << item.nbHParent() << "'>";
        //   trace->info() << "Truc : " << item.nbHParent();
        //   for (Integer j = 0; j < item.nbHParent(); ++j) {
        //     ofile << ' ' << item.parent(j).uniqueId();
        //   }
        //   ofile << "</parent>";
        // }
        {
          ofile << "<child count='" << cell.nbHChildren() << "'>";
          for (Integer j = 0; j < cell.nbHChildren(); ++j) {
            ofile << ' ' << cell.hChild(j).uniqueId();
          }
          ofile << "</child>";
        }
        ofile << "</amr>";
      }
      {
        ghost_cells_layer1.clear();
        for (Node node : cell.nodes()) {
          for (Cell sub_cell : node.cells()) {
            ghost_cells_layer1.add(sub_cell.uniqueId().asInt64());
          }
        }

        {
          // Trie la liste des mailles fantômes et retire les doublons.
          std::sort(std::begin(ghost_cells_layer1), std::end(ghost_cells_layer1));
          auto new_end = std::unique(std::begin(ghost_cells_layer1), std::end(ghost_cells_layer1));
          ghost_cells_layer1.resize(arcaneCheckArraySize(new_end - std::begin(ghost_cells_layer1)));
          ofile << "<ghost1 count='" << ghost_cells_layer1.size() << "'>";
          for (auto j : ghost_cells_layer1)
            ofile << ' ' << j;
          ofile << "</ghost1>\n";
        }
      }
      ofile << "</cell>\n";
    }
    ofile << "</cells>\n";
  }

  // Sauve les groupes

  {
    ofile << "<groups>\n";
    // Trie les groupes par ordre alphabétique pour être certains qu'ils sont
    // toujours écrits dans le même ordre.
    std::map<String,ItemGroup> sorted_groups;
    for (ItemGroupCollection::Enumerator i_group(mesh->groups()); ++i_group;) {
      const ItemGroup& group = *i_group;
      if (group.isLocalToSubDomain())
        continue;
      sorted_groups.insert(std::make_pair(group.name(),group));
    }
    for ( const auto& [name,group] : sorted_groups ){
      ofile << "<group name='" << group.name()
            << "' kind='" << itemKindName(group.itemKind())
            << "' count='" << group.size() << "'>\n";
      ENUMERATE_ (Item, i_item, group) {
        ofile << ' ' << i_item->uniqueId();
      }
      ofile << "\n</group>\n";
    }
    ofile << "</groups>\n";
  }

  // Sauve les interfaces liées
  {
    ofile << "<tied-interfaces>\n";
    TiedInterfaceCollection tied_interfaces(mesh->tiedInterfaces());
    for (TiedInterfaceCollection::Enumerator itied(tied_interfaces); ++itied;) {
      ITiedInterface* interface = *itied;
      FaceGroup slave_group = interface->slaveInterface();
      FaceGroup master_group = interface->masterInterface();
      ofile << "<tied-interface master_name='" << master_group.name()
            << "' slave_name='" << slave_group.name() << "'>\n";
      TiedInterfaceNodeList tied_nodes(interface->tiedNodes());
      TiedInterfaceFaceList tied_faces(interface->tiedFaces());
      ENUMERATE_FACE (iface, master_group) {
        Face face = *iface;
        //FaceVectorView slave_faces = face.slaveFaces();
        ofile << "<master-face uid='" << face.uniqueId() << "'>\n";
        for (Integer zz = 0, zs = tied_nodes[iface.index()].size(); zz < zs; ++zz) {
          TiedNode tn = tied_nodes[iface.index()][zz];
          ofile << "<node uid='" << tn.node().uniqueId() << "' iso='" << tn.isoCoordinates() << "' />\n";
        }
        for (Integer zz = 0, zs = tied_faces[iface.index()].size(); zz < zs; ++zz) {
          TiedFace tf = tied_faces[iface.index()][zz];
          ofile << "<face uid='" << tf.face().uniqueId() << "'/>\n";
        }
        ofile << "</master-face>\n";
      }
      ofile << "</tied-interface>\n";
    }
    ofile << "</tied-interfaces>\n";
  }
  ofile << "</mesh-connectivity>\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshUtilsCheckConnectivity
{
 public:

  class ItemInternalXml
  {
   public:

    ItemInternalXml() {}
    explicit ItemInternalXml(Item item)
    : m_item(item)
    {}

   public:

    bool operator<(const ItemInternalXml& i2) const
    {
      return m_item.uniqueId() < i2.m_item.uniqueId();
    }
    bool operator<(Int64 uid) const
    {
      return m_item.uniqueId() < uid;
    }

   public:

    Item m_item;
    XmlNode m_element;
  };
  typedef std::map<Int64, ItemInternalXml> ItemInternalXmlMap;

 public:

  MeshUtilsCheckConnectivity(IMesh* mesh, const XmlNode& doc_node, bool check_sub_domain)
  : m_mesh(mesh)
  , m_doc_node(doc_node)
  , m_has_error(false)
  , m_check_sub_domain(check_sub_domain)
  {
  }

 public:

  void doCheck();

 public:

  IMesh* m_mesh;
  XmlNode m_doc_node;
  bool m_has_error;
  bool m_check_sub_domain;
  ItemInternalXmlMap m_nodes_internal;
  ItemInternalXmlMap m_edges_internal;
  ItemInternalXmlMap m_faces_internal;
  ItemInternalXmlMap m_cells_internal;
  Int64UniqueArray m_items_unique_id;

 private:

  /*void _sortByUniqueIds(eItemKind ik,Array<ItemInternalXml>& items_internal)
    {
      ItemGroup all_items(m_mesh->allItems(ik));
      items_internal.resize(all_items.size());
      
      Integer index = 0;
      ENUMERATE_ITEM(i,all_items){
        const Item& item = *i;
        items_internal[index].m_item = item.internal();
        ++index;
      }
      std::sort(items_internal.begin(),items_internal.end());
      }*/
  void _read(eItemKind ik, ItemInternalXmlMap& items_internal,
             XmlNode root_node, const String& lower_case_kind_name);

  /*ItemInternalXml* _find(Array<ItemInternalXml>& items_internal,Integer uid)
    {
      ItemInternalXmlIterator z = std::lower_bound(items_internal.begin(),items_internal.end(),uid);
      if (z==items_internal.end())
        return 0;
      ItemInternalXml* ixml = &(*z);
      if (ixml->m_item->uniqueId()!=uid)
        return 0;
      return ixml;
      //cout << "NOT IMPLEMENTED!\n";
      //return 0;
      }*/
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
checkMeshConnectivity(IMesh* mesh, const XmlNode& doc_node, bool check_sub_domain)
{
  if (!mesh)
    return;
  MeshUtilsCheckConnectivity v(mesh, doc_node, check_sub_domain);
  v.doCheck();
}

void MeshUtils::
checkMeshConnectivity(IMesh* mesh, const String& file_name, bool check_sub_domain)
{
  ITraceMng* tm = mesh->traceMng();
  ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromFile(file_name, tm));
  //IXmlDocumentHolder* doc = io_mng->parseXmlFile(file_name);
  XmlNode root = doc->documentNode();
  checkMeshConnectivity(mesh, root, check_sub_domain);
}

void MeshUtilsCheckConnectivity::
_read(eItemKind ik, ItemInternalXmlMap& items_internal, XmlNode root_node,
      const String& lower_case_kind_name)
{
  ITraceMng* trace = m_mesh->traceMng();

  //_sortByUniqueIds(ik,items_internal);

  String ustr_uid("uid");
  String ustr_cells("cells");
  String ustr_count("count");
  String ustr_nodes("nodes");
  String ustr_ghost1("ghost1");

  String kind_name(itemKindName(ik));

  ENUMERATE_ITEM (iitem, m_mesh->itemFamily(ik)->allItems()) {
    Item item = *iitem;
    ItemInternalXml ixml(item);
    items_internal.insert(ItemInternalXmlMap::value_type(item.uniqueId().asInt64(), ixml));
  }

#ifdef ARCANE_DEBUG_MESH
  for (Integer i = 0; i < items_internal.size(); ++i) {
    const ItemInternalXml& item_xml = items_internal[i];
    trace->info() << "Item " << kind_name << ":" << item_xml.m_item->uniqueId()
                  << ' ' << i << ' ' << item_xml.m_item;
  }
#endif

  XmlNodeList xml_items(root_node.children(lower_case_kind_name));
  for (const auto& xml_node : xml_items) {
    Integer uid = xml_node.attr(ustr_uid).valueAsInteger();
    ItemInternalXmlMap::iterator iixml = items_internal.find(uid);
    if (iixml != items_internal.end()) {
      iixml->second.m_element = xml_node;
    }
#if 0
    ItemInternalXml* ixml = _find(items_internal,uid);
    if (ixml){
      ixml->m_element = xml_node;
#ifdef ARCANE_DEBUG_MESH
      trace->info() << "FOUND " << uid << ' ' << z->m_item->uniqueId() << ' ' << z->m_element.name()
                  << ' ' << z->m_item;
#endif
    }
#ifdef ARCANE_DEBUG_MESH
    else
      trace->info() << "FOUND " << uid << " NOT HERE";
#endif
#endif
  }

  Int32UniqueArray local_ids;
  local_ids.reserve(100);
  for (ItemInternalXmlMap::const_iterator i(items_internal.begin()); i != items_internal.end(); ++i) {
    const ItemInternalXml& item_xml = i->second;
    Item item = item_xml.m_item;
    const XmlNode& xitem = item_xml.m_element;
    if (xitem.null()) {
      trace->error() << "Item " << kind_name << ":" << item.uniqueId()
                     << "unknown in reference mesh";
      m_has_error = true;
      continue;
    }
    if (ik != IK_Node) {
      ItemWithNodes item_with_node(item);
      XmlNode xitem_node = xitem.child(ustr_nodes);
      Integer ref_nb_node = xitem_node.attr(ustr_count).valueAsInteger();
      Integer nb_node = item_with_node.nbNode();
      if (ref_nb_node != nb_node) {
        trace->error() << "Item " << kind_name << ":" << item.uniqueId()
                       << ": number of nodes (" << nb_node << ") "
                       << "different than reference (" << ref_nb_node << ")";
        m_has_error = true;
        continue;
      }

      m_items_unique_id.reserve(ref_nb_node);
      String s = xitem_node.value();
      _stringToIds(s, m_items_unique_id);
      bool is_bad = false;
      for (NodeEnumerator i_node(item_with_node.nodes()); i_node(); ++i_node) {
        if (m_items_unique_id[i_node.index()] != i_node->uniqueId()) {
          is_bad = true;
          break;
        }
      }
      if (is_bad) {
        m_has_error = true;
        OStringStream ostr;
        ostr() << "Item " << kind_name << ":" << item.uniqueId()
               << ": nodes (";
        for (NodeEnumerator i_node(item_with_node.nodes()); i_node(); ++i_node) {
          ostr() << ' ' << i_node->uniqueId();
        }
        ostr() << ") different than reference (" << s << ")";
        trace->error() << ostr.str();
      }
    }
    if (item.isOwn()) {
      // Si c'est une maille, recherche si les mailles qui doivent être
      // fantômes sont bien présentes dans le maillage.
      // Si c'est un noeud, recherche si les mailles autour de ce noeud
      // sont bien présentes dans le maillage.
      XmlNode elem;
      if (ik == IK_Cell)
        elem = xitem.child(ustr_ghost1);
      else if (ik == IK_Node)
        elem = xitem.child(ustr_cells);
      if (!elem.null()) {
        _stringToIds(elem.value(), m_items_unique_id);
        local_ids.resize(m_items_unique_id.size());
        m_mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids, m_items_unique_id, false);
        StringBuilder not_found;
        bool has_not_found = false;
        for (Integer uui = 0, uuis = m_items_unique_id.size(); uui < uuis; ++uui) {
          if (local_ids[uui] == NULL_ITEM_ID) {
            not_found += " ";
            not_found += m_items_unique_id[uui];
            m_has_error = true;
            has_not_found = true;
          }
        }
        if (has_not_found) {
          if (ik == IK_Cell)
            trace->info() << "ERROR: One or more ghost cells of cell "
                          << ItemPrinter(item) << " are not in the sub-domain"
                          << " ref='" << elem.value() << "' not_found='" << not_found << '\'';
          else if (ik == IK_Node) {
            trace->info() << "ERROR: One or more cells with node "
                          << ItemPrinter(item) << " are not in the sub-domain"
                          << " ref='" << elem.value() << "' not_found='" << not_found << '\'';
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtilsCheckConnectivity::
doCheck()
{
  ITraceMng* trace = m_mesh->traceMng();

  trace->info() << "Checking mesh checkMeshConnectivity()";

  XmlNode root_node = m_doc_node.child(String("mesh-connectivity"));
  if (!root_node) {
    trace->warning() << "Incorrect connectivity file";
    return;
  }
  XmlNode nodes_root = root_node.child("nodes");
  XmlNode faces_root = root_node.child("faces");
  XmlNode cells_root = root_node.child("cells");

  _read(IK_Node, m_nodes_internal, nodes_root, "node");
  _read(IK_Face, m_faces_internal, faces_root, "face");
  _read(IK_Cell, m_cells_internal, cells_root, "cell");

  String ustr_groups("groups");
  String ustr_group("group");
  String ustr_count("count");

  XmlNode groups_root = root_node.child(ustr_groups);
  for (ItemGroupCollection::Enumerator i_group(m_mesh->groups()); ++i_group;) {
    const ItemGroup& group = *i_group;
    if (group.isLocalToSubDomain() || group.isOwn())
      continue;
    XmlNode group_elem = groups_root.childWithNameAttr(ustr_group, String(group.name()));
    if (group_elem.null()) {
      m_has_error = true;
      trace->error() << "Unable to find group <" << group.name()
                     << "> in reference file";
      continue;
    }
    Integer size = group_elem.attr(ustr_count).valueAsInteger();
    m_items_unique_id.reserve(size);
    _stringToIds(group_elem.value(), m_items_unique_id);
    Integer ref_size = m_items_unique_id.size();
    if (ref_size != size) {
      trace->error() << "Number of items in group <" << group.name()
                     << "> (" << size << " different than reference (" << ref_size;
    }
    // TODO: vérifier que les toutes les entités du groupe dans le maillage
    // de référence sont aussi dans le groupe correspondant de ce maillage.
  }

  if (m_has_error) {
    trace->fatal() << "Error(s) while checking mesh";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
checkMeshProperties(IMesh* mesh, bool is_sorted, bool has_no_hole, bool check_faces)
{
  if (!mesh)
    return;
  if (has_no_hole)
    throw NotImplementedException(A_FUNCINFO, "Has no hole");

  ITraceMng* trace = mesh->traceMng();
  bool has_error = false;
  if (is_sorted) {
    //for( Integer iki=0; iki<NB_ITEM_KIND; ++iki ){
    //  eItemKind ik = static_cast<eItemKind>(iki);
    for (IItemFamilyCollection::Enumerator i(mesh->itemFamilies()); ++i;) {
      eItemKind ik = (*i)->itemKind();
      if (!check_faces && ik == IK_Face)
        continue;
      // Il est normal que les particules ne soient pas triées
      if (ik == IK_Particle)
        continue;
      //ItemGroup all_items = mesh->itemFamily(ik)->allItems();
      ItemGroup all_items = (*i)->allItems();
      Item last_item;
      ENUMERATE_ITEM (iitem, all_items) {
        Item item = *iitem;

        if (!last_item.null() && (last_item.uniqueId() >= item.uniqueId() || last_item.localId() >= item.localId())) {
          trace->error() << "Item not sorted " << ItemPrinter(item, ik)
                         << " Last item " << ItemPrinter(last_item, ik);
          has_error = true;
        }
        last_item = item;
      }
    }
  }
  if (has_error) {
    ARCANE_FATAL("Missing required mesh properties (sorted and/or no hole)");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
printItems(std::ostream& ostr, const String& name, ItemGroup item_group)
{
  ostr << " ------- " << name << '\n';
  ENUMERATE_ITEM (iitem, item_group) {
    ostr << FullItemPrinter((*iitem)) << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshUtils::
reorderNodesOfFace(Int64ConstArrayView before_ids, Int64ArrayView after_ids)
{
  // \a true s'il faut réorienter les faces pour que leur orientation
  // soit indépendante du partitionnement du maillage initial.
  bool need_swap_orientation = false;
  Integer min_node_index = 0;
  Integer nb_node = before_ids.size();

  // Traite directement le cas des arêtes
  if (nb_node == 2) {
    if (before_ids[0] < before_ids[1]) {
      after_ids[0] = before_ids[0];
      after_ids[1] = before_ids[1];
      return false;
    }
    after_ids[0] = before_ids[1];
    after_ids[1] = before_ids[0];
    return true;
  }

  // L'algorithme suivant oriente les faces en tenant compte uniquement
  // de l'ordre de la numérotation de ces noeuds. Si cet ordre est
  // conservé lors du partitionnement, alors l'orientation des faces
  // sera aussi conservée.

  // L'algorithme est le suivant:
  // - Recherche le noeud n de plus petit indice.
  // - Recherche n-1 et n+1 les indices de ses 2 noeuds voisins.
  // - Si (n+1) est inférieur à (n-1), l'orientation n'est pas modifiée.
  // - Si (n+1) est supérieur à (n-1), l'orientation est inversée.

  // Recherche le noeud de plus petit indice

  Int64 min_node = INT64_MAX;
  for (Integer k = 0; k < nb_node; ++k) {
    Int64 id = before_ids[k];
    if (id < min_node) {
      min_node = id;
      min_node_index = k;
    }
  }
  Int64 next_node = before_ids[(min_node_index + 1) % nb_node];
  Int64 prev_node = before_ids[(min_node_index + (nb_node - 1)) % nb_node];
  Integer incr = 0 ;
  Integer incr2 = 0 ;
  if(next_node==min_node)
  {
    next_node = before_ids[(min_node_index + (nb_node + 2)) % nb_node];
    incr = 1 ;
  }
  if(prev_node==min_node)
  {
    prev_node = before_ids[(min_node_index + (nb_node - 2)) % nb_node];
    incr2 = nb_node - 1 ;
  }
  if (next_node > prev_node)
    need_swap_orientation = true;
  if (need_swap_orientation) {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (nb_node - k + min_node_index + incr) % nb_node;
      after_ids[k] = before_ids[index];
    }
  }
  else {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (k + min_node_index + incr2) % nb_node;
      after_ids[k] = before_ids[index];
    }
  }
  return need_swap_orientation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshUtils::
reorderNodesOfFace2(Int64ConstArrayView nodes_unique_id, IntegerArrayView new_index)
{
  // \a true s'il faut réorienter les faces pour que leur orientation
  // soit indépendante du partitionnement du maillage initial.
  bool need_swap_orientation = false;
  Integer min_node_index = 0;
  Integer nb_node = nodes_unique_id.size();

  // Traite directement le cas des arêtes
  if (nb_node == 2) {
    if (nodes_unique_id[0] < nodes_unique_id[1]) {
      new_index[0] = 0;
      new_index[1] = 1;
      return false;
    }
    new_index[0] = 1;
    new_index[1] = 0;
    return true;
  }

  // L'algorithme suivant oriente les faces en tenant compte uniquement
  // de l'ordre de la numérotation de ces noeuds. Si cet ordre est
  // conservé lors du partitionnement, alors l'orientation des faces
  // sera aussi conservée.

  // L'algorithme est le suivant:
  // - Recherche le noeud n de plus petit indice.
  // - Recherche n-1 et n+1 les indices de ses 2 noeuds voisins.
  // - Si (n+1) est inférieur à (n-1), l'orientation n'est pas modifiée.
  // - Si (n+1) est supérieur à (n-1), l'orientation est inversée.

  // Recherche le noeud de plus petit indice

  Int64 min_node = INT64_MAX;
  for (Integer k = 0; k < nb_node; ++k) {
    Int64 id = nodes_unique_id[k];
    if (id < min_node) {
      min_node = id;
      min_node_index = k;
    }
  }
  Int64 next_node = nodes_unique_id[(min_node_index + 1) % nb_node];
  Int64 prev_node = nodes_unique_id[(min_node_index + (nb_node - 1)) % nb_node];
  Integer incr = 0 ;
  Integer incr2 = 0 ;
  if(next_node==min_node)
  {
    next_node = nodes_unique_id[(min_node_index + 2) % nb_node];
    incr = 1 ;
  }
  if(prev_node==min_node)
  {
    prev_node = nodes_unique_id[(min_node_index + (nb_node - 2)) % nb_node];
    incr2 = nb_node - 1 ;
  }
  if (next_node > prev_node)
    need_swap_orientation = true;
  if (need_swap_orientation) {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (nb_node - k + min_node_index + incr) % nb_node;
      new_index[k] = index;
    }
  }
  else {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (k + min_node_index + incr2) % nb_node;
      new_index[k] = index;
    }
  }
  return need_swap_orientation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Face MeshUtils::
getFaceFromNodesLocalId(Node node, Int32ConstArrayView face_nodes_local_id)
{
  Integer n = face_nodes_local_id.size();
  for (Integer i = 0, s = node.nbFace(); i < s; ++i) {
    Face f(node.face(i));
    Integer fn = f.nbNode();
    if (fn == n) {
      bool same_face = true;
      for (Integer zz = 0; zz < n; ++zz)
        if (f.node(zz).localId() != face_nodes_local_id[zz]) {
          same_face = false;
          break;
        }
      if (same_face)
        return f;
    }
  }
  return Face();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Face MeshUtils::
getFaceFromNodesUniqueId(Node node, Int64ConstArrayView face_nodes_unique_id)
{
  Integer n = face_nodes_unique_id.size();
  for (Integer i = 0, s = node.nbFace(); i < s; ++i) {
    Face f(node.face(i));
    Integer fn = f.nbNode();
    if (fn == n) {
      bool same_face = true;
      for (Integer zz = 0; zz < n; ++zz)
        if (f.node(zz).uniqueId() != face_nodes_unique_id[zz]) {
          same_face = false;
          break;
        }
      if (same_face)
        return f;
    }
  }
  return Face();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
removeItemAndKeepOrder(Int32ArrayView items, Int32 local_id)
{
  Integer n = items.size();
  if (n <= 0)
    ARCANE_FATAL("Can not remove item lid={0} because list is empty", local_id);

  --n;
  if (n == 0) {
    if (items[0] == local_id)
      return;
  }
  else {
    // Si l'élément est le dernier, ne fait rien.
    if (items[n] == local_id)
      return;
    for (Integer i = 0; i < n; ++i) {
      if (items[i] == local_id) {
        for (Integer z = i; z < n; ++z)
          items[z] = items[z + 1];
        return;
      }
    }
  }
  // TODO: Il faut activer cela mais pour l'instant cela fait planter un test.
  //ARCANE_FATAL("No entity with local_id={0} found in list {1}",local_id,items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
shrinkMeshGroups(IMesh* mesh)
{
  auto f = [&](ItemGroup& group) {
    group.internal()->shrinkMemory();
  };
  meshvisitor::visitGroups(mesh, f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MeshUtils::
printMeshGroupsMemoryUsage(IMesh* mesh, Int32 print_level)
{
  ITraceMng* tm = mesh->traceMng();
  Int64 total_capacity = 0;
  Int64 total_computed_capacity = 0;
  auto f = [&](ItemGroup& group) {
    ItemGroupImpl* p = group.internal();
    // Attention à bien prendre la taille du groupe via \a p
    // car sinon pour un groupe calculé on le reconstruit.
    Int64 c = p->capacity();
    bool is_computed = p->hasComputeFunctor();
    total_capacity += c;
    if (is_computed)
      total_computed_capacity += c;
    if (print_level >= 1)
      tm->info() << "GROUP Name=" << group.name() << " computed?=" << p->hasComputeFunctor()
                 << " nb_ref=" << p->nbRef() << " size=" << p->size()
                 << " capacity=" << c;
  };
  meshvisitor::visitGroups(mesh, f);
  tm->info() << "MeshGroupsMemoryUsage: capacity = " << total_capacity
             << " computed_capacity=" << total_computed_capacity;
  return total_capacity * sizeof(Int32);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
dumpSynchronizerTopologyJSON(IVariableSynchronizer* var_syncer, const String& filename)
{
  IParallelMng* pm = var_syncer->parallelMng();
  ITraceMng* tm = pm->traceMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32ConstArrayView comm_ranks = var_syncer->communicatingRanks();

  tm->info(4) << "Dumping VariableSynchronizerTopology filename=" << filename;
  Int32 nb_comm_rank = comm_ranks.size();

  UniqueArray<Int32> nb_items_by_rank(nb_comm_rank);
  for (Integer i = 0; i < nb_comm_rank; ++i)
    nb_items_by_rank[i] = var_syncer->sharedItems(i).size();

  JSONWriter json_writer(JSONWriter::FormatFlags::None);
  json_writer.beginObject();

  if (my_rank == 0) {
    UniqueArray<Int32> all_nb_comm_ranks(nb_rank);
    pm->gather(Int32ConstArrayView(1, &nb_comm_rank), all_nb_comm_ranks, 0);
    json_writer.write("NbNeighbor", all_nb_comm_ranks);

    {
      UniqueArray<Int32> all_neighbor_ranks;
      pm->gatherVariable(comm_ranks, all_neighbor_ranks, 0);
      json_writer.write("NeighborsRank", all_neighbor_ranks);
    }
    {
      UniqueArray<Int32> all_nb_items_by_rank;
      pm->gatherVariable(nb_items_by_rank, all_nb_items_by_rank, 0);
      json_writer.write("NeighborsSize", all_nb_items_by_rank);
    }
  }
  else {
    pm->gather(Int32ConstArrayView(1, &nb_comm_rank), {}, 0);
    UniqueArray<Int32> empty_array;
    pm->gatherVariable(comm_ranks, empty_array, 0);
    pm->gatherVariable(nb_items_by_rank, empty_array, 0);
  }

  json_writer.endObject();

  if (my_rank == 0) {
    std::ofstream ofile(filename.localstr());
    auto bytes = json_writer.getBuffer().bytes();
    ofile.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  class MyIdsToTest
  {
   public:

    friend bool operator<(const MyIdsToTest& a, const MyIdsToTest& b)
    {
      return a.ids < b.ids;
    }
    static constexpr int MAX_SIZE = 16;

   public:

    std::array<Int32, MAX_SIZE> ids = {};
  };

  void _computePatternOccurence(const ItemGroup& items, const String& message,
                                IndexedItemConnectivityViewBase2 cty)
  {
    std::map<MyIdsToTest, Int32> occurence_map;
    Int32 nb_skipped = 0;
    ENUMERATE_ (Item, iitem, items) {
      Item item = *iitem;
      MyIdsToTest diff_ids;

      Int32 index = 0;
      Int32 lid0 = 0;
      bool is_skipped = false;
      for (ItemLocalId sub_item : cty.items(item)) {
        if (index >= MyIdsToTest::MAX_SIZE) {
          is_skipped = true;
          break;
        }
        if (index == 0)
          lid0 = sub_item.localId();
        diff_ids.ids[index] = sub_item.localId() - lid0;
        //info() << "  Cell lid=" << item.localId() << " I=" << index << " diff_lid=" << diff_ids.ids[index];
        ++index;
      }
      if (is_skipped)
        ++nb_skipped;
      else
        ++occurence_map[diff_ids];
    }
    ITraceMng* tm = items.mesh()->traceMng();
    tm->info() << "Occurence: " << message << " group=" << items.name()
               << " nb=" << items.size() << " map_size=" << occurence_map.size()
               << " nb_skipped=" << nb_skipped;
  }
} // namespace

void MeshUtils::
computeConnectivityPatternOccurence(IMesh* mesh)
{
  ARCANE_CHECK_POINTER(mesh);

  UnstructuredMeshConnectivityView cty(mesh);

  _computePatternOccurence(mesh->allNodes(), "NodeCells", cty.nodeCell());
  _computePatternOccurence(mesh->allNodes(), "NodeFaces", cty.nodeFace());
  _computePatternOccurence(mesh->allFaces(), "FaceCells", cty.faceCell());
  _computePatternOccurence(mesh->allFaces(), "FaceNodes", cty.faceNode());
  _computePatternOccurence(mesh->allCells(), "CellNodes", cty.cellNode());
  _computePatternOccurence(mesh->allCells(), "CellFaces", cty.cellFace());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
markMeshConnectivitiesAsMostlyReadOnly(IMesh* mesh, RunQueue* queue, bool do_prefetch)
{
  if (!mesh)
    return;
  IVariableMng* vm = mesh->variableMng();
  VariableCollection used_variables = vm->usedVariables();
  const String tag_name = "ArcaneConnectivity";
  DataAllocationInfo alloc_info(eMemoryLocationHint::HostAndDeviceMostlyRead);

  // Les variables associées à la connectivité ont le tag 'ArcaneConnectivity'.
  for (VariableCollection::Enumerator iv(used_variables); ++iv;) {
    IVariable* v = *iv;
    if (!v->hasTag(tag_name))
      continue;
    if (v->meshHandle().meshOrNull() == mesh) {
      v->setAllocationInfo(alloc_info);
      if (do_prefetch)
        VariableUtils::prefetchVariableAsync(v, queue);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ItemBase MeshUtils::
findOneItem(IItemFamily* family, Int64 unique_id)
{
  ItemInternal* v = family->findOneItem(unique_id);
  if (v)
    return { v };
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ItemBase MeshUtils::
findOneItem(IItemFamily* family, ItemUniqueId unique_id)
{
  ItemInternal* v = family->findOneItem(unique_id);
  if (v)
    return { v };
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MeshUtils::impl
{
  //! Retourne le max des uniqueId() des entités de \a group
  Int64 _getMaxUniqueId(const ItemGroup& group, Int64 max_uid)
  {
    ENUMERATE_ (Item, iitem, group) {
      Item item = *iitem;
      if (max_uid < item.uniqueId())
        max_uid = item.uniqueId();
    }
    return max_uid;
  }
} // namespace MeshUtils::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemUniqueId MeshUtils::
getMaxItemUniqueIdCollective(IMesh* mesh)
{
  Int64 max_uid = NULL_ITEM_UNIQUE_ID;
  max_uid = impl::_getMaxUniqueId(mesh->allNodes(), max_uid);
  max_uid = impl::_getMaxUniqueId(mesh->allEdges(), max_uid);
  max_uid = impl::_getMaxUniqueId(mesh->allCells(), max_uid);
  max_uid = impl::_getMaxUniqueId(mesh->allFaces(), max_uid);
  Int64 global_max = mesh->parallelMng()->reduce(Parallel::ReduceMax, max_uid);
  return ItemUniqueId(global_max);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
checkUniqueIdsHashCollective(IItemFamily* family, IHashAlgorithm* hash_algo,
                             const String& expected_hash, bool print_hash,
                             bool include_ghost)
{
  ARCANE_CHECK_POINTER(family);
  ARCANE_CHECK_POINTER(hash_algo);

  IParallelMng* pm = family->parallelMng();
  ITraceMng* tm = family->traceMng();

  UniqueArray<Int64> own_items_uid;
  ItemGroup own_items_group = (include_ghost ? family->allItems() : family->allItems().own());
  ENUMERATE_ (Item, iitem, own_items_group) {
    Item item{ *iitem };
    own_items_uid.add(item.uniqueId());
  }
  UniqueArray<Int64> global_items_uid;
  pm->allGatherVariable(own_items_uid, global_items_uid);
  std::sort(global_items_uid.begin(), global_items_uid.end());

  UniqueArray<Byte> hash_result;
  hash_algo->computeHash64(asBytes(global_items_uid.constSpan()), hash_result);
  String hash_str = Convert::toHexaString(hash_result);
  if (print_hash)
    tm->info() << "HASH_RESULT family=" << family->name()
               << " v=" << hash_str << " expected=" << expected_hash;
  if (!expected_hash.empty() && hash_str != expected_hash)
    ARCANE_FATAL("Bad hash for uniqueId() for family '{0}' v={1} expected='{2}'",
                 family->fullName(), hash_str, expected_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
fillUniqueIds(ItemVectorView items,Array<Int64>& uids)
{
  Integer nb_item = items.size();
  uids.resize(nb_item);
  ENUMERATE_ITEM (iitem, items)
    uids[iitem.index()] = iitem->uniqueId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MeshUtils::
generateHashUniqueId(SmallSpan<const Int64> nodes_unique_id)
{
  // Tout les bits sont formées avec la fonction de hash.
  // Le uniqueId() généré doit toujours être positif
  // sauf pour l'entité nulle.
  Int32 nb_node = nodes_unique_id.size();
  if (nb_node == 0)
    return -1;
  using Hasher = IntegerHashFunctionT<Int64>;
  Int64 uid0 = nodes_unique_id[0];
  Int64 hash = Hasher::hashfunc(uid0);
  for (Int32 i = 1; i < nb_node; ++i) {
    Int64 next_hash = Hasher::hashfunc(nodes_unique_id[i]);
    hash ^= next_hash + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  }
  Int64 new_uid = abs(hash);
  ARCANE_ASSERT(new_uid >= 0, ("UniqueId is not >= 0"));
  return new_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
computeAndSetOwnerForNodes(IMesh* mesh)
{
  ARCANE_CHECK_POINTER(mesh);
  mesh->utilities()->computeAndSetOwnersForNodes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
computeAndSetOwnerForEdges(IMesh* mesh)
{
  ARCANE_CHECK_POINTER(mesh);
  mesh->utilities()->computeAndSetOwnersForEdges();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUtils::
computeAndSetOwnerForFaces(IMesh* mesh)
{
  ARCANE_CHECK_POINTER(mesh);
  mesh->utilities()->computeAndSetOwnersForFaces();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
