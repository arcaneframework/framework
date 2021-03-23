// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TiedInterfaceMng.h                                          (C) 2000-2010 */
/*                                                                           */
/* Gestionnaire des interfaces liées.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_TIEDINTERFACEMNG_H
#define ARCANE_MESH_TIEDINTERFACEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/List.h"

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class XmlNode;
class IMeshPartitionConstraint;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class TiedInterface;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation du gestionnaire d'interfaces liées.
 */
class TiedInterfaceMng
: public TraceAccessor
{
 public:

  TiedInterfaceMng(DynamicMesh* mesh);
  virtual ~TiedInterfaceMng();

 public:
  
  void computeTiedInterfaces(const XmlNode& mesh_node);
  void prepareTiedInterfacesForDump();
  void readTiedInterfacesFromDump();
  bool hasTiedInterface()
  {
    return !m_tied_interfaces.empty();
  }
  TiedInterfaceCollection tiedInterfaces()
  {
    return m_tied_interfaces;
  }

  ISubDomain* subDomain() { return m_sub_domain; }
  ConstArrayView<TiedInterface*> trueTiedInterfaces()
  {
    return m_true_tied_interfaces;
  }

 private:

  DynamicMesh* m_mesh;
  ISubDomain* m_sub_domain;
  String m_name;

 private:

  TiedInterfaceList m_tied_interfaces;
  VariableArrayInt64 m_tied_interface_items_info;
  VariableArrayReal2 m_tied_interface_nodes_iso;
  VariableArrayString m_tied_interface_face_groups;
  UniqueArray<TiedInterface*> m_true_tied_interfaces;
  IMeshPartitionConstraint* m_tied_constraint;

 private:
  
  void _deleteTiedInterfaces();
  void _applyTiedInterfaceStructuration(TiedInterface* tied_interface);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
