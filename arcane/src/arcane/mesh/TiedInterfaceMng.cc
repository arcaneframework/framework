// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceMng.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des interfaces liées.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/TiedInterfaceMng.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/FaceFamily.h"

#include "arcane/mesh/TiedInterface.h"

#include "arcane/XmlNode.h"
#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"
#include "arcane/ICaseDocument.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMeshPartitionConstraintMng.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/ArcaneException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceMng::
TiedInterfaceMng(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_sub_domain(mesh->subDomain())
, m_name(mesh->name())
, m_tied_interface_items_info(VariableBuildInfo(m_sub_domain,m_name+"TiedInterfaceItemsInfo"))
, m_tied_interface_nodes_iso(VariableBuildInfo(m_sub_domain,m_name+"TiedInterfaceNodesIso"))
, m_tied_interface_face_groups(VariableBuildInfo(m_sub_domain,m_name+"TiedInterfaceFaceGroups"))
, m_tied_constraint(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TiedInterfaceMng::
~TiedInterfaceMng()
{
  // Le m_tied_constraint est détruit par le gestionnaire de contraintes du maillage
  m_tied_constraint = 0;

  _deleteTiedInterfaces();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceMng::
computeTiedInterfaces(const XmlNode& mesh_node)
{
  ICaseDocument* doc = subDomain()->caseDocument();
  CaseNodeNames* cnn = doc->caseNodeNames();
  String tied_interfaces_name = cnn->tied_interfaces;
  String tied_interface_not_structured_name = cnn->tied_interfaces_not_structured;
  String tied_interfaces_planar_tolerance = cnn->tied_interfaces_planar_tolerance;
  String semi_conform_name = cnn->tied_interfaces_semi_conform;
  String name_attribute = cnn->tied_interfaces_slave;
  XmlNode tied_interface_elem = mesh_node.child(tied_interfaces_name);
  XmlNodeList child_list = tied_interface_elem.children(semi_conform_name);
  bool has_error = false;

  IParallelMng* pm = m_mesh->parallelMng();
  bool is_parallel = pm->isParallel();
  UniqueArray<FaceGroup> interfaces_group;
  UniqueArray<bool> is_structured_list;
  UniqueArray<Real> planar_tolerance_list;
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  for( auto& i : child_list ){
    XmlNode group_attr = i.attr(name_attribute);
    if (group_attr.null()){
      error() << "Attribute '" << name_attribute << "' missing";
      has_error = true;
      continue;
    }
    String group_name = i.attrValue(name_attribute);
    FaceGroup face_group = face_family.findGroup(group_name);
    if (face_group.null()){
      error() << "Can't find the interface '" << group_name << "'";
      has_error = true;
      continue;
    }
    if (interfaces_group.contains(face_group)){
      error() << "The group '" << group_name << "' is already present in list of tied interfaces.";
      has_error = true;
      continue;
    }
    interfaces_group.add(face_group);

    bool is_not_structured = i.attr(tied_interface_not_structured_name).valueAsBoolean();
    info() << "** NOT STRUCTURED? = " << is_not_structured;
    is_structured_list.add(!is_not_structured);
    XmlNode tolerance_node = i.attr(tied_interfaces_planar_tolerance);
    if (tolerance_node.null()) {
       planar_tolerance_list.add(0.);
    } else {
       planar_tolerance_list.add(tolerance_node.valueAsReal());
    }
  }
  if (has_error)
    fatal() << "Can't determine the tied interfaces";

  // S'il n'y a pas d'interface liée spécifié dans le jeu de données,
  // recherche si un groupe de face de nom SOUDURE ou SOUDURES
  // existe et dans ce cas le considère comme une interface de soudure
  {
    FaceGroup g1 = face_family.findGroup("SOUDURE");
    if (g1.null())
      g1 = face_family.findGroup("SOUDURES");
    if (!g1.null()){
      // N'ajoute le groupe que s'il n'est pas déjà dans la liste
      if (!interfaces_group.contains(g1)){
        info() << "Add automatically the group '" << g1.name() << "' to the list of tied interfaces";
        interfaces_group.add(g1);
        is_structured_list.add(true);
        planar_tolerance_list.add(0.0);
      }
      else{
        info() << "built-in group " << g1.name() << " already in list of tied interfaces";
      }
    }
  }
  Integer nb_interface = interfaces_group.size();
  IItemFamily* cell_family = m_mesh->cellFamily();
  if (nb_interface!=0){
    // En parallèle, il faudra migrer des mailles pour que les mailles
    // de part et d'autre d'une face de l'interface soient dans le même sous-domaine.
    // En parallèle, il faut d'abord recalculer les propriétaire, faire
    // l'échange puis enfin calculer les projections
    if (is_parallel){
      VariableItemInt32& cells_owner = cell_family->itemsNewOwner();
      ENUMERATE_ITEM(iitem,cell_family->allItems()){
        cells_owner[iitem] = (*iitem).owner();
      }
      //for( Integer i=0, is=interfaces_group.size(); i<is; ++i ){
        //FaceGroup face_group = interfaces_group[i];
      TiedInterface::PartitionConstraintBase* c = TiedInterface::createConstraint(m_mesh,interfaces_group);
      m_tied_constraint = c;
      IMeshPartitionConstraintMng* pcmng = m_mesh->partitionConstraintMng();
      pcmng->addConstraint(c);
      pcmng->computeAndApplyConstraints();
      //}
      cells_owner.synchronize();
      m_mesh->utilities()->changeOwnersFromCells();
      m_mesh->setDynamic(true);
      m_mesh->exchangeItems();
      // Indique que le partitionnement initial a été effectué
      c->setInitialRepartition(false);
    }
    for( Integer i=0, is=interfaces_group.size(); i<is; ++i ){
      FaceGroup face_group = interfaces_group[i];
      info() << " Semi-conform interface name=" << face_group.name();
      TiedInterface* tied_interface = new TiedInterface(m_mesh);
      if (planar_tolerance_list[i] > 0)
        tied_interface->setPlanarTolerance(planar_tolerance_list[i]);
      tied_interface->build(face_group,is_structured_list[i]);
      //if (is_structured_list[i]){
      //_applyTiedInterfaceStructuration(tied_interface);
      //}
      m_true_tied_interfaces.add(tied_interface);
      m_tied_interfaces.add(tied_interface);
      face_family.applyTiedInterface(tied_interface);
    }

    face_family.endUpdate();
    face_family.prepareForDump();
  }
  prepareTiedInterfacesForDump();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceMng::
prepareTiedInterfacesForDump()
{
  Integer nb_tied_interface = m_tied_interfaces.count();
  m_tied_interface_face_groups.resize(nb_tied_interface*2);
  
  Integer total_nb_node = 0;
  Integer total_nb_face = 0;
  Integer nb_tied_node_per_interface = 0;
  Integer nb_tied_face_per_interface = 0;
  for( Integer i=0; i<nb_tied_interface; ++i ){
  	ITiedInterface* ti = m_tied_interfaces[i];

    TiedInterfaceNodeList nodes = ti->tiedNodes();
    nb_tied_node_per_interface += nodes.dim1Size();
    for( Integer zz=0, zs=nodes.dim1Size(); zz<zs; ++zz )
      total_nb_node += nodes[zz].size();

    TiedInterfaceFaceList faces = ti->tiedFaces();
    nb_tied_face_per_interface += faces.dim1Size();
    for( Integer zz=0, zs=faces.dim1Size(); zz<zs; ++zz )
      total_nb_face += faces[zz].size();
  }

  Integer items_info_size = total_nb_node + total_nb_face +
  nb_tied_node_per_interface + nb_tied_face_per_interface + nb_tied_interface*2;

  Integer nodes_iso_size = total_nb_node;

  m_tied_interface_items_info.resize(items_info_size);
  m_tied_interface_nodes_iso.resize(nodes_iso_size);

  Integer items_info_index = 0;
  Integer nodes_iso_index = 0;
  for( Integer i=0; i<nb_tied_interface; ++i ){
    ITiedInterface* ti = m_tied_interfaces[i];

    TiedInterfaceNodeList nodes = ti->tiedNodes();
    Integer nb_node = nodes.dim1Size();
    m_tied_interface_items_info[items_info_index] = nb_node;
    ++items_info_index;
    for( Integer zz=0; zz<nb_node; ++zz ){
      Integer nb_node2 = nodes[zz].size();
      m_tied_interface_items_info[items_info_index] = nb_node2;
      ++items_info_index;
    }

    for( Integer zz=0; zz<nb_node; ++zz ){
      Integer nb_node2 = nodes[zz].size();
      for( Integer zz2=0; zz2<nb_node2; ++zz2 ){
        m_tied_interface_items_info[items_info_index] = nodes[zz][zz2].node().uniqueId().asInt64();
        ++items_info_index;
        m_tied_interface_nodes_iso[nodes_iso_index] = nodes[zz][zz2].isoCoordinates();
        ++nodes_iso_index;
      }
    }

    TiedInterfaceFaceList faces = ti->tiedFaces();
    Integer nb_face = faces.dim1Size();
    m_tied_interface_items_info[items_info_index] = nb_face;
    ++items_info_index;
    for( Integer zz=0; zz<nb_face; ++zz ){
      Integer nb_face2 = faces[zz].size();
      m_tied_interface_items_info[items_info_index] = nb_face2;
      ++items_info_index;
    }

    for( Integer zz=0; zz<nb_face; ++zz ){
      Integer nb_face2 = faces[zz].size();
      for( Integer zz2=0; zz2<nb_face2; ++zz2 ){
        m_tied_interface_items_info[items_info_index] = faces[zz][zz2].face().uniqueId().asInt64();
        ++items_info_index;
      }
    }
  }
  debug() << "ITEMS_INFO_INDEX=" << items_info_index << " N=" << items_info_size;
  debug() << "NODES_ISO_INDEX=" << nodes_iso_index << " N=" << nodes_iso_size;
  if (nodes_iso_index!=nodes_iso_size)
    throw InternalErrorException(A_FUNCINFO,"Bad size for 'nodes_iso_index'");
  if (items_info_index!=items_info_size)
    throw InternalErrorException(A_FUNCINFO,"Bad size for 'items_info_index'");
  for( Integer i=0; i<nb_tied_interface; ++i ){
    ITiedInterface* ti = m_tied_interfaces[i];
    m_tied_interface_face_groups[(i*2)] = ti->masterInterfaceName();
    m_tied_interface_face_groups[(i*2)+1] = ti->slaveInterfaceName();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceMng::
readTiedInterfacesFromDump()
{
  _deleteTiedInterfaces();
  Integer nb_tied_interface = m_tied_interface_face_groups.size() / 2;

  Integer items_info_index = 0;
  Integer nodes_iso_index = 0;

  IItemFamily* node_family = m_mesh->nodeFamily();
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  NodeInfoListView nodes_internal(node_family);
  FaceInfoListView faces_internal(&face_family);
  UniqueArray<TiedNode> nodes;
  UniqueArray<TiedFace> faces;

  for( Integer i=0; i<nb_tied_interface; ++i ){
    TiedInterface* tied_interface = new TiedInterface(m_mesh);
    String master_group_name = m_tied_interface_face_groups[(i*2)];
    String slave_group_name = m_tied_interface_face_groups[(i*2)+1];
    tied_interface->reload(&face_family,master_group_name,slave_group_name);
    FaceGroup master_group = tied_interface->masterInterface();
    FaceGroup slave_group = tied_interface->slaveInterface();
    m_tied_interfaces.add(tied_interface);
    m_true_tied_interfaces.add(tied_interface);

    Integer nb_node = arcaneCheckArraySize(m_tied_interface_items_info[items_info_index]);
    ++items_info_index;
    IntegerUniqueArray items_size(nb_node);
    for( Integer zz=0; zz<nb_node; ++zz ){
      Integer nb_node2 = arcaneCheckArraySize(m_tied_interface_items_info[items_info_index]);
      ++items_info_index;
      items_size[zz] = nb_node2;
    }
    tied_interface->resizeNodes(items_size);
    Int32UniqueArray local_ids;
    for( Integer zz=0; zz<nb_node; ++zz ){
      Integer nb_node2 = items_size[zz];
      nodes.clear();
      local_ids.resize(nb_node2);
      Int64ConstArrayView unique_ids(nb_node2,&m_tied_interface_items_info[items_info_index]);
      node_family->itemsUniqueIdToLocalId(local_ids,unique_ids,true);
      items_info_index += nb_node2;

      for( Integer zz2=0; zz2<nb_node2; ++zz2 ){
        Node node(nodes_internal[local_ids[zz2]]);
        nodes.add(TiedNode(zz2,node,m_tied_interface_nodes_iso[nodes_iso_index]));
        ++nodes_iso_index;
      }
      tied_interface->setNodes(zz,nodes);
    }

    Integer nb_face = arcaneCheckArraySize(m_tied_interface_items_info[items_info_index]);
    ++items_info_index;
    items_size.resize(nb_face);
    for( Integer zz=0; zz<nb_face; ++zz ){
      Integer nb_face2 = arcaneCheckArraySize(m_tied_interface_items_info[items_info_index]);
      ++items_info_index;
      items_size[zz] = nb_face2;
    }
    tied_interface->resizeFaces(items_size);
    UniqueArray<TiedFace> faces2;
    for( Integer zz=0; zz<nb_face; ++zz ){
      Integer nb_face2 = items_size[zz];
      faces2.clear();
      local_ids.resize(nb_face2);
      Int64ConstArrayView unique_ids(nb_face2,&m_tied_interface_items_info[items_info_index]);
      face_family.itemsUniqueIdToLocalId(local_ids,unique_ids,true);
      items_info_index += nb_face2;
      for( Integer zz2=0; zz2<nb_face2; ++zz2 ){
        Face face(faces_internal[local_ids[zz2]]);
        faces2.add(TiedFace(zz2,face));
      }
      tied_interface->setFaces(zz,faces2);
    }
    info() << "Read interface nb_face=" << nb_face << " nb_node=" << nb_node;
  }

  // Reconstruit les contraintes si necessaire
  if (!m_tied_constraint){
    info() << "Rebuilding tied interface constraints";
    UniqueArray<FaceGroup> interface_groups;
    for( Integer i=0, n=m_true_tied_interfaces.size(); i<n; ++i )
      interface_groups.add(m_true_tied_interfaces[i]->slaveInterface());

    Integer nb_interface = interface_groups.size();
    if (nb_interface!=0){
      TiedInterface::PartitionConstraintBase* c = TiedInterface::createConstraint(m_mesh,interface_groups);
      m_tied_constraint = c;
      IMeshPartitionConstraintMng* pcmng = m_mesh->partitionConstraintMng();
      pcmng->addConstraint(c);
      // Indique que le partitionnement initial a été effectué
      c->setInitialRepartition(false);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceMng::
_applyTiedInterfaceStructuration(TiedInterface* tied_interface)
{
  // Suppose que l'interface est structurée MxN.
  // Pour chercher M, parcours la liste des noeuds liées et détecte
  // quand un noeud à une coordonnées iso.y inférieure à celle du
  // noeud précédent. Cela signifie qu'on change de ligne dans
  // la structuration.
  // Une fois MxN connu, normalise les coordonnées iso pour
  // qu'elles correspondent à cette structuration

  TiedInterfaceNodeList nodes = tied_interface->tiedNodes();
  for( Integer zz=0, zs=nodes.dim1Size(); zz<zs; ++zz ){
    Integer current_nb_node = nodes[zz].size();
    Real old_y = -10.0;
    Integer computed_m = 0;
    for( Integer i=0; i<current_nb_node; ++i ){
      Real2 iso_val = nodes[zz][i].isoCoordinates();
      info() << "ISO zz=" << zz << " i=" << i << " v=" << iso_val << " old=" << old_y;
      if (iso_val.y>old_y){
        ++computed_m;
      }
      else{
        info() << "COMPUTED M = "<< computed_m;
        computed_m = 0;
      }
      old_y = iso_val.y;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TiedInterfaceMng::
_deleteTiedInterfaces()
{
  for( Integer i=0, is=m_tied_interfaces.count(); i<is; ++i )
    delete m_tied_interfaces[i];
  m_tied_interfaces.clear();
  m_true_tied_interfaces.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
