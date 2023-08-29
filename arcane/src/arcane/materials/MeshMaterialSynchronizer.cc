// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialSynchronizer.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Synchronisation des entités des matériaux.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/MeshMaterialSynchronizer.h"

#include "arcane/VariableTypes.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"

#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialModifier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSynchronizer::
MeshMaterialSynchronizer(IMeshMaterialMng* material_mng)
: TraceAccessor(material_mng->traceMng())
, m_material_mng(material_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialSynchronizer::
~MeshMaterialSynchronizer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void MeshMaterialSynchronizer::
_setBit(ByteArrayView bytes,Integer position)
{
  Integer offset = position / 8;
  Integer bit = position % 8;
  bytes[offset] |= (Byte)(1 << bit);
}

inline bool MeshMaterialSynchronizer::
_hasBit(ByteConstArrayView bytes,Integer position)
{
  Integer offset = position / 8;
  Integer bit = position % 8;
  return bytes[offset] & (1 << bit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSynchronizer::
_fillPresence(AllEnvCell all_env_cell,ByteArrayView presence)
{
  ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
    ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
      MatCell mc = *imatcell;
      Integer mat_index = mc.materialId();
      _setBit(presence,mat_index);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialSynchronizer::
synchronizeMaterialsInCells()
{
  /*
    L'algorithme utilisé est le suivant:
    
    On utilise une variable aux mailles qui utilise un bit pour chaque
    matériau pour indiquer sa présence: si ce bit est positionné, le matériau
    est présent, sinon il est absent. La variable utilisée est donc de type
    ArrayByte aux mailles. Les méthodes _hasBit() et _setBit() permettent
    de positionner le bit d'un matériau donné.

    1. Le sous-domaine remplit cette variables pour ces mailles.
    2. La variable est synchronisée.
    3. Le sous-domaine compare pour chacune de ses mailles fantômes
    ce tableau de présence des matériaux et ajoute/supprime les matériaux en fonction
    de ce tableau.
  */
  IMesh* mesh = m_material_mng->mesh();
  if (!mesh->parallelMng()->isParallel())
    return false;

  ConstArrayView<IMeshMaterial*> materials = m_material_mng->materials();
  Integer nb_mat = materials.size();
  VariableCellArrayByte mat_presence(VariableBuildInfo(mesh,"ArcaneMaterialSyncPresence"));
  Integer dim2_size = nb_mat / 8;
  if ((nb_mat % 8)!=0)
    ++dim2_size;
  mat_presence.resize(dim2_size);
  info(4) << "Resize presence variable nb_mat=" << nb_mat << " dim2=" << dim2_size;
  CellToAllEnvCellConverter cell_converter = m_material_mng->cellToAllEnvCellConverter();
  ENUMERATE_CELL(icell,mesh->ownCells()){
    ByteArrayView presence = mat_presence[icell];
    presence.fill(0);
    AllEnvCell all_env_cell = cell_converter[*icell];
    _fillPresence(all_env_cell,presence);
  }

  bool has_changed = false;

  mat_presence.synchronize();
  {
    ByteUniqueArray before_presence(dim2_size);
    UniqueArray< UniqueArray<Int32> > to_add(nb_mat);
    UniqueArray< UniqueArray<Int32> > to_remove(nb_mat);
    ENUMERATE_CELL(icell,mesh->allCells()){
      Cell cell = *icell;
      // Ne traite que les mailles fantomes.
      if (cell.isOwn())
        continue;
      Int32 cell_lid = cell.localId();
      AllEnvCell all_env_cell = cell_converter[cell];
      before_presence.fill(0);
      _fillPresence(all_env_cell,before_presence);
      ByteConstArrayView after_presence = mat_presence[cell];
      // Ajoute/Supprime cette maille des matériaux si besoin.
      for( Integer imat=0; imat<nb_mat; ++imat ){
        bool has_before = _hasBit(before_presence,imat);
        bool has_after = _hasBit(after_presence,imat);
        if (has_before && !has_after){
          to_remove[imat].add(cell_lid);
        }
        else if (has_after && !has_before)
          to_add[imat].add(cell_lid);
      }
    }

    MeshMaterialModifier modifier(m_material_mng);
    for( Integer i=0; i<nb_mat; ++i ){
      if (!to_add[i].empty()){
        modifier.addCells(materials[i],to_add[i]);
        has_changed = true;
      }
      if (!to_remove[i].empty()){
        modifier.removeCells(materials[i],to_remove[i]);
        has_changed = true;
      }
    }
  }

  return has_changed;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les mailles des matériaux sont bien cohérentes entre les
 * sous-domaines.
 * Cette méthode est collective
 */
void MeshMaterialSynchronizer::
checkMaterialsInCells(Integer max_print)
{
  /*
    Pour cela, on utilise une variable aux mailles et on applique
    l'algorithme suivant pour chaque matériau:
    - le sous-domaine propriétaire remplit cette variable
    avec l'indice du matériau
    - la variable est synchronisée.
    - chaque sous-domaine vérifie ensuite pour chaque maille
    que si la variable a pour valeur l'indice du matériau, alors
    ce matériau est présent.
  */
 
  IMesh* mesh = m_material_mng->mesh();
  if (!mesh->parallelMng()->isParallel())
    return;
  m_material_mng->checkValid();
  info(4) << "CheckMaterialsInCells";
  VariableCellInt32 indexes(VariableBuildInfo(mesh,"ArcaneMaterialPresenceIndexes"));
  _checkComponents(indexes,m_material_mng->materialsAsComponents(),max_print);
  _checkComponents(indexes,m_material_mng->environmentsAsComponents(),max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialSynchronizer::
_checkComponents(VariableCellInt32& indexes,
                 ConstArrayView<IMeshComponent*> components,
                 Integer max_print)
{
  IMesh* mesh = m_material_mng->mesh();
  Integer nb_component = components.size();
  Integer nb_error = 0;

  info() << "Checking components nb=" << nb_component;

  CellToAllEnvCellConverter all_env_cell_converter(m_material_mng);

  for( Integer i=0; i<nb_component; ++i ){
    indexes.fill(-1);
    IMeshComponent* c = components[i];
    ENUMERATE_COMPONENTCELL(iccell,c){
      ComponentCell cc = *iccell;
      indexes[cc.globalCell()] = i;
    }

    indexes.synchronize();

    ENUMERATE_ALLENVCELL(iallenvcell,m_material_mng,mesh->allCells()){
      AllEnvCell all_env_cell = *iallenvcell;
      Cell cell = all_env_cell.globalCell();
      bool has_sync_mat = (indexes[cell]==i);
      ComponentCell cc = c->findComponentCell(all_env_cell);
      bool has_component = !cc.null();
      if (has_sync_mat!=has_component){
        ++nb_error;
        if (max_print<0 || nb_error<max_print)
          error() << "Bad component synchronisation for i=" << i
                  << " name=" << c->name()
                  << " cell_uid=" << cell.uniqueId()
                  << " sync_mat=" << has_sync_mat
                  << " has_component=" << has_component;
      }
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad synchronisation");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
