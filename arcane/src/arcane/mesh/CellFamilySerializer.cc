// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellFamilySerializer.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Sérialisation/Désérialisation des familles de mailles.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/CellFamilySerializer.h"

#include "arcane/core/ISerializer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/MeshPartInfo.h"

#include "arcane/mesh/FullItemInfo.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/OneMeshItemAdder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellFamilySerializer::
CellFamilySerializer(CellFamily* family,bool use_flags,
                     DynamicMeshIncrementalBuilder* mesh_builder)
: TraceAccessor(family->traceMng())
, m_mesh_builder(mesh_builder)
, m_family(family)
, m_use_flags(use_flags)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamilySerializer::
serializeItems(ISerializer* buf,Int32ConstArrayView cells_local_id)
{
  ItemInternalList cells_internal = m_family->itemsInternal();

  const Integer nb_cell = cells_local_id.size();

  IMesh* mesh = m_family->mesh();
  Int32 my_rank = mesh->meshPartInfo().partRank();
  const Integer parent_info = FullCellInfo::parentInfo(mesh);
  const bool has_amr = mesh->isAmrActivated();
  const bool has_edge = m_mesh_builder->hasEdge();
  const bool use_flags = m_use_flags;

  info(4) << "_serializeItems : "
          << mesh->name() << " "
          << m_family->name() << " "
          << m_family->parentFamilyDepth();

  switch(buf->mode()){
  case ISerializer::ModeReserve:
    {
      buf->reserveInt64(2); // 1 pour le rang du sous-domaine et 1 pour le nombre de mailles
      UniqueArray<Int64> tmp_buf;
      tmp_buf.reserve(200);
      for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
        Int32 lid = cells_local_id[i_cell];
        ItemInternal* cell = cells_internal[lid];
        tmp_buf.clear();
        FullCellInfo::dump(cell,tmp_buf,parent_info,has_edge,has_amr,use_flags);
        buf->reserveInt64(1);
        buf->reserveSpan(tmp_buf);
      }
    }
    break;
  case ISerializer::ModePut:
    {
      buf->putInt64(my_rank); // Stocke le numéro du sous-domaine
      buf->putInt64(nb_cell); // Stocke le nombre de mailles
      info(4) <<  "Serialize: Put: nb_cell=" << nb_cell;
      UniqueArray<Int64> tmp_buf;
      tmp_buf.reserve(200);
      for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
        Int32 lid = cells_local_id[i_cell];
        ItemInternal* cell = cells_internal[lid];
        tmp_buf.clear();
        FullCellInfo::dump(cell,tmp_buf,parent_info,has_edge,has_amr,use_flags);
        buf->putInt64(tmp_buf.largeSize());
        buf->putSpan(tmp_buf);
      }
    }
    break;
  case ISerializer::ModeGet:
    {
      deserializeItems(buf,nullptr);
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellFamilySerializer::
deserializeItems(ISerializer* buf,Int32Array* cells_local_id)
{
  IMesh* mesh = m_family->mesh();
  ItemTypeMng* itm = mesh->itemTypeMng();
  Int32 my_rank = mesh->meshPartInfo().partRank();
  Int32 orig_rank = CheckedConvert::toInt32(buf->getInt64()); // Numéro du sous-domaine source
  Int64 nb_cell = buf->getInt64();
  const Integer parent_info = FullCellInfo::parentInfo(mesh);

  Int64UniqueArray mem_buf;
  mem_buf.reserve(200);
  Int32ArrayView locals_id_view;
  bool use_array = false;
  if (cells_local_id){
    cells_local_id->resize(nb_cell);
    locals_id_view = (*cells_local_id);
    use_array = true;
  }
  const bool is_check = arcaneIsCheck();
  const bool has_edge = m_mesh_builder->hasEdge();
  const bool has_amr = mesh->isAmrActivated();
  const bool use_flags = m_use_flags;
  info(4) << "DeserializeCells: nb_cell=" << nb_cell << " orig=" << orig_rank << " has_edge=" << has_edge
          << " has_amr=" << has_amr << " use_flags=" << use_flags;
  for( Integer i_cell=0; i_cell<nb_cell; ++i_cell ){
    Int64 memory_needed = buf->getInt64();
    mem_buf.resize(memory_needed);
    buf->getSpan(mem_buf);
    FullCellInfo current_cell(mem_buf,0,itm,parent_info,has_edge,has_amr,use_flags);
    ItemInternal* icell = m_mesh_builder->oneMeshItemAdder()->addOneCell(current_cell);
    Cell cell(icell);
    if (use_array)
      locals_id_view[i_cell] = icell->localId();
    {
      // Attention dans le flag IT_Own n'est pas correct, il est forcé apres
      if (use_flags)
        icell->setFlags(current_cell.flags()) ;
      // Force les owners à la valeur de sérialisation, car ils ne seront
      // pas corrects dans le cas ou la maille ou un de ses éléments
      // existait déjà
      icell->setOwner(current_cell.owner(),my_rank);
      if (is_check){
        // Vérifie que les unique_ids des éléments actuels de la maille
        // sont les même que ceux qu'on désérialise.
        bool has_error = false;
        Integer node_index = 0;
        for( Node node : cell.nodes() ){
          has_error |= (node.uniqueId()!=current_cell.nodeUniqueId(node_index));
          ++node_index;
        }
        Integer edge_index = 0;
        for( Edge edge : cell.edges() ){
          has_error |= (edge.uniqueId()!=current_cell.edgeUniqueId(edge_index));
          ++edge_index;
        }
        Integer face_index = 0;
        for( Face face : cell.faces() ){
          has_error |= (face.uniqueId()!=current_cell.faceUniqueId(face_index));
          ++face_index;
        }
        if (has_error){
          node_index = 0;
          for( Node node : cell.nodes() ){
            info() << "Cell c=" << ItemPrinter(cell) << " node=" << ItemPrinter(node)
                   << " remote_uid=" << current_cell.nodeUniqueId(node_index);
            ++node_index;
          }
          edge_index = 0;
          for( Edge edge : cell.edges() ){
            info() << "Cell c=" << ItemPrinter(cell) << " edge=" << ItemPrinter(edge)
                   << " remote_uid=" << current_cell.edgeUniqueId(edge_index);
            ++edge_index;
          }
          face_index = 0;
          for( Face face : cell.faces() ){
            info() << "Cell c=" << ItemPrinter(cell) << " face=" << ItemPrinter(face)
                   << " remote_uid=" << current_cell.faceUniqueId(face_index);
            ++face_index;
          }
          ARCANE_FATAL("Incoherent local and remote node, edge or face unique id");
        }
      }

      // TODO: vérifier si cela est utile. A priori non car le travail
      // est fait lors de la désérialisation des mailles.
      Integer node_index = 0;
      for( Node node : cell.nodes() ){
        node.mutableItemBase().setOwner(current_cell.nodeOwner(node_index),my_rank);
        ++node_index;
      }
      Integer edge_index = 0;
      for( Edge edge : cell.edges() ){
        edge.mutableItemBase().setOwner(current_cell.edgeOwner(edge_index),my_rank);
        ++edge_index;
      }
      Integer face_index = 0;
      for( Face face : cell.faces() ){
        face.mutableItemBase().setOwner(current_cell.faceOwner(face_index),my_rank);
        ++face_index;
      }
    }
  }
  info(4) << "EndDeserializeCells: nb_cell=" << nb_cell << " orig=" << orig_rank;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* CellFamilySerializer::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
