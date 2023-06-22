// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5ItemVariableInfo.cc                                     (C) 2000-2023 */
/*                                                                           */
/* Lecture de variables au format HDF5.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/hdf5/Hdf5VariableInfoBase.h"

#include "arcane/core/MeshVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"

#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType>
class Hdf5ItemVariableInfo
: public Hdf5VariableInfoBase
{
 public:
  Hdf5ItemVariableInfo(IMesh* mesh,IVariable* v);
 public:
  VariableType& trueVariable() { return m_variable; }
  virtual IVariable* variable() const { return m_variable.variable(); }
 public:
  virtual void readVariable(Hdf5Utils::HFile& hfile,const String& filename,
                            Hdf5Utils::StandardTypes& st,const String& ids_hpath,IData* data);
  virtual void writeVariable(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st);

 private:
  void _readStandardArray(Array<DataType>& buffer,Array<Int64>& unique_ids,const String& ids_hpath,
                          hid_t file_id,Hdf5Utils::StandardTypes& st);
  void _writeStandardArray(Array<DataType>& buffer,hid_t file_id,Hdf5Utils::StandardTypes& st);
 private:
  VariableType m_variable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType>
class Hdf5ScalarVariableInfo
: public Hdf5VariableInfoBase
{
 public:
  Hdf5ScalarVariableInfo(IVariable* v);
 public:
  VariableType& trueVariable() { return m_variable; }
  virtual IVariable* variable() const { return m_variable.variable(); }
 public:
  virtual void readVariable(Hdf5Utils::HFile& hfile,const String& filename,
                            Hdf5Utils::StandardTypes& st,const String& ids_hpath,
                            IData* data);
  virtual void writeVariable(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st);

 private:
  void _readStandardArray(Array<DataType>& buffer,
                          hid_t file_id,Hdf5Utils::StandardTypes& st);
  void _writeStandardArray(Array<DataType>& buffer,hid_t file_id,Hdf5Utils::StandardTypes& st);
 private:
  VariableType m_variable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableInfoBase* Hdf5VariableInfoBase::
create(IMesh* mesh,const String& name,const String& family_name)
{
  IItemFamily* family = mesh->findItemFamily(family_name,true);
  IVariable* var = family->findVariable(name,true);
  return Hdf5VariableInfoBase::create(var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5VariableInfoBase* Hdf5VariableInfoBase::
create(IVariable* var)
{
  _checkValidVariable(var);
  Hdf5VariableInfoBase* var_info = 0;
  IItemFamily* item_family = var->itemFamily();
  if (item_family){
    IMesh* mesh = item_family->mesh();
    if (var->isPartial()){
      switch(var->dataType()){
      case DT_Real:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemReal,Real>(mesh,var);
        break;
      case DT_Real2:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemReal2,Real2>(mesh,var);
        break;
      case DT_Real2x2:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemReal2x2,Real2x2>(mesh,var);
        break;
      case DT_Real3:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemReal3,Real3>(mesh,var);
        break;
      case DT_Real3x3:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemReal3x3,Real3x3>(mesh,var);
        break;
      case DT_Byte:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemByte,Byte>(mesh,var);
        break;
      case DT_Int32:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemInt32,Int32>(mesh,var);
        break;
      case DT_Int64:
        var_info = new Hdf5ItemVariableInfo<PartialVariableItemInt64,Int64>(mesh,var);
        break;
      default:
        throw FatalErrorException(A_FUNCINFO,"Bad variable type");
        break;
      }
    }
    else{
      switch(var->dataType()){
      case DT_Real:
        var_info = new Hdf5ItemVariableInfo<VariableItemReal,Real>(mesh,var);
        break;
      case DT_Real2:
        var_info = new Hdf5ItemVariableInfo<VariableItemReal2,Real2>(mesh,var);
        break;
      case DT_Real2x2:
        var_info = new Hdf5ItemVariableInfo<VariableItemReal2x2,Real2x2>(mesh,var);
        break;
      case DT_Real3:
        var_info = new Hdf5ItemVariableInfo<VariableItemReal3,Real3>(mesh,var);
        break;
      case DT_Real3x3:
        var_info = new Hdf5ItemVariableInfo<VariableItemReal3x3,Real3x3>(mesh,var);
        break;
      case DT_Byte:
        var_info = new Hdf5ItemVariableInfo<VariableItemByte,Byte>(mesh,var);
        break;
      case DT_Int32:
        var_info = new Hdf5ItemVariableInfo<VariableItemInt32,Int32>(mesh,var);
        break;
      case DT_Int64:
        var_info = new Hdf5ItemVariableInfo<VariableItemInt64,Int64>(mesh,var);
        break;
      default:
        throw FatalErrorException(A_FUNCINFO,"Bad variable type");
        break;
      }
    }
  }
  else{
    if (var->dimension()==0){
      switch(var->dataType()){
      case DT_Real:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarReal,Real>(var);
        break;
      case DT_Real2:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarReal2,Real2>(var);
        break;
      case DT_Real2x2:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarReal2x2,Real2x2>(var);
        break;
      case DT_Real3:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarReal3,Real3>(var);
        break;
      case DT_Real3x3:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarReal3x3,Real3x3>(var);
        break;
      case DT_Byte:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarByte,Byte>(var);
        break;
      case DT_Int32:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarInt32,Int32>(var);
        break;
      case DT_Int64:
        var_info = new Hdf5ScalarVariableInfo<VariableScalarInt64,Int64>(var);
        break;
      default:
        throw FatalErrorException(A_FUNCINFO,"Bad variable type");
        break;
      }
    }
  }
  if (!var_info)
    throw NotSupportedException(A_FUNCINFO,
                                String::format("IData for variable '{0}'",var->fullName()));
  return var_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableInfoBase::
_checkValidVariable(IVariable* var)
{
  IItemFamily* item_family = var->itemFamily();
  if (item_family){
    if (var->dimension()==1)
      return;
  }
  else{
    if (var->dimension()==0)
      return;
  }

  ARCANE_FATAL("Bad variable '{0}'. Variable has to be an item variable and have dimension"
               "'1' or be a scalar variable",var->fullName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableInfoBase::
writeGroup(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st,
           const String& hdf_path,Integer save_type)
{
  IVariable* var = variable();
  ItemGroup group = var->itemGroup();
  IParallelMng* pm = group.mesh()->parallelMng();
  ItemGroup enumerate_group = group.own();
  Integer nb_item = enumerate_group.size();
  ITraceMng* tm = pm->traceMng();

  // Pour l'instant la méthode parallèle créé un tampon
  // du nombre total d'éléments du tableau
  ///TODO a optimiser
  Int64UniqueArray unique_ids(nb_item);
  {
    Integer index = 0;
    ENUMERATE_ITEM(iitem,enumerate_group){
      unique_ids[index] = (*iitem).uniqueId();
      ++index;
    }
  }

  if (save_type & SAVE_IDS){
    Hdf5Utils::StandardArrayT<Int64> ids_writer(hfile.id(),hdf_path+"_Ids");
    ids_writer.parallelWrite(pm,st,unique_ids,unique_ids);
  }

  // Pour l'instant, on ne peut sauver que des tableaux 2D dont le nombre
  // d'éléments dans la 2ème dimension est identique. Cela pose problème
  // si toutes les entités n'ont pas le même nombre de noeuds. Pour
  // éviter ce problème, on calcule le nombre max de noeud possible et on
  // utilise cette valeur. Pour les entités qui ont moins de noeuds, on
  // ajoute comme coordonnées des NaN. Cela n'est pas optimum notamment
  // si une seule entité a beaucoup plus de noeuds que les autres mais
  // cela fonctionne dans tous les cas.
  if (save_type & SAVE_COORDS){
    IMesh* mesh = enumerate_group.mesh();
    VariableNodeReal3& nodes_coords(mesh->toPrimaryMesh()->nodesCoordinates());
    eItemKind item_kind = enumerate_group.itemKind();
    if (item_kind==IK_Edge || item_kind==IK_Face || item_kind==IK_Cell){
      Integer index = 0;
      Real3UniqueArray coords;
      Real3UniqueArray centers;
      Integer max_nb_node = 0;
      const Real nan_value = std::numeric_limits<Real>::quiet_NaN();
      const Real3 real3_nan = Real3(nan_value,nan_value,nan_value);
      {
        ENUMERATE_ITEMWITHNODES(iitem,enumerate_group){
          ItemWithNodes item = (*iitem).toItemWithNodes();
          Integer nb_node = item.nbNode();
          if (nb_node>max_nb_node)
            max_nb_node = nb_node;
        }
        max_nb_node = pm->reduce(Parallel::ReduceMax,max_nb_node);
      }
      Int32UniqueArray items_type(nb_item);
      ENUMERATE_ITEMWITHNODES(iitem,enumerate_group){
        ItemWithNodes item = (*iitem).toItemWithNodes();
        Integer nb_node = item.nbNode();
        Real3 item_center;
        for( NodeLocalId inode : item.nodeIds() ){
          coords.add(nodes_coords[inode]);
          item_center += nodes_coords[inode];
        }
        item_center /= nb_node;
        // Ajoute des NaN pour les coordonnées restantes
        for( Integer k=nb_node; k<max_nb_node; ++k )
          coords.add(real3_nan);
        centers.add(item_center);
        items_type[index] = item.typeInfo()->typeId();
        ++index;
      }
      Hdf5Utils::StandardArrayT<Real3> coords_writer(hfile.id(),hdf_path+"_Coords");
      coords_writer.parallelWrite(pm,st,coords,unique_ids);
      Hdf5Utils::StandardArrayT<Real3> centers_writer(hfile.id(),hdf_path+"_Center");
      centers_writer.parallelWrite(pm,st,centers,unique_ids);
      Hdf5Utils::StandardArrayT<Int32> types_writer(hfile.id(),hdf_path+"_Types");
      types_writer.parallelWrite(pm,st,items_type,unique_ids);
    }
    else if (item_kind==IK_Node){
      Real3UniqueArray coords;
      ENUMERATE_NODE(iitem,enumerate_group){
        coords.add(nodes_coords[iitem]);
      }
      Hdf5Utils::StandardArrayT<Real3> coords_writer(hfile.id(),hdf_path+"_Center");
      coords_writer.parallelWrite(pm,st,coords,unique_ids);
    }
    else
      tm->pwarning() << "Can not save coordinates for family name="
                     << enumerate_group.itemFamily()->name();
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5VariableInfoBase::
readGroupInfo(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st,
              const String& hdf_path,Int64Array& uids,Real3Array& centers)
{
  IVariable* var = variable();
  ItemGroup group = var->itemGroup();
  IParallelMng* pm = group.mesh()->parallelMng();
  ItemGroup enumerate_group = group;
  ITraceMng* tm = pm->traceMng();
  bool is_master = pm->isMasterIO();

  // Lit les unique ids sauvegardés
  Int64UniqueArray dummy_uids;
  Int64UniqueArray saved_unique_ids;
  Hdf5Utils::StandardArrayT<Int64> ids_reader(hfile.id(),hdf_path+"_Ids");
  Integer nb_old_item = 0;
  if (is_master){
    ids_reader.readDim();
    nb_old_item = arcaneCheckArraySize(ids_reader.dimensions()[0]);
    tm->info() << "NB_OLD_ITEM nb=" << nb_old_item;
    saved_unique_ids.resize(nb_old_item);
    dummy_uids.resize(nb_old_item);
  }
  ids_reader.parallelRead(pm,st,saved_unique_ids,dummy_uids);

  Real3UniqueArray saved_centers(nb_old_item);
  Hdf5Utils::StandardArrayT<Real3> centers_reader(hfile.id(),hdf_path+"_Center");
  if (is_master)
    centers_reader.readDim();
  centers_reader.parallelRead(pm,st,saved_centers,dummy_uids);
  tm->info() << "READ SAVED CENTERS nb=" << saved_centers.size();

  uids.copy(saved_unique_ids);
  centers.copy(saved_centers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType>
Hdf5ItemVariableInfo<VariableType,DataType>::
Hdf5ItemVariableInfo(IMesh* mesh,IVariable* v)
: Hdf5VariableInfoBase()
, m_variable(v)
{
  ARCANE_UNUSED(mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ItemVariableInfo<VariableType,DataType>::
readVariable(Hdf5Utils::HFile& hfile,const String& filename,
             Hdf5Utils::StandardTypes& st,const String& ids_hpath,IData* var_data)
{
  ARCANE_UNUSED(ids_hpath);

  if (!var_data)
    var_data = variable()->data();
  if (!var_data)
    throw ArgumentException(A_FUNCINFO,"Null var_data");
  UniqueArray<DataType> buffer;
  IArrayDataT<DataType>* data_array = dynamic_cast<IArrayDataT<DataType>*>(var_data);
  if (!data_array){
    const char* n = typeid(var_data).name();
    throw FatalErrorException(A_FUNCINFO,String::format("Bad type for IData '{0}'",n));
  }
  ArrayView<DataType> var_value = data_array->view();
  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  ITraceMng* tm = vm->traceMng();
  IParallelMng* pm = vm->parallelMng();
  bool is_master = pm->isMasterIO();
  //Integer master_rank = pm->masterIORank();
  if (is_master){
    if (hfile.id()<0)
      hfile.openRead(filename);
  }

  Int64UniqueArray unique_ids;
  _readStandardArray(buffer,unique_ids,ids_hpath,hfile.id(),st);
  
  Integer buf_size = buffer.size();
  //ArrayView<DataType> var_value = m_variable.asArray();
  Integer nb_var_value = var_value.size();
  if (var->isPartial()){
    // Dans le cas d'une variables partielle, il faut d'abord
    // mettre dans une table de hashage la valeur pour chaque uid,
    // puis parcourir le groupe de la variable et remplir
    // la valeur pour chaque uid.
    HashTableMapT<Int64,DataType> values_from_uid(buf_size,true);
    for( Integer z=0; z<buf_size; ++z ){
      values_from_uid.add(unique_ids[z],buffer[z]);
    }
    ItemGroup var_group = m_variable.itemGroup();
    ENUMERATE_ITEM(iitem,var_group){
      Item item = *iitem;
      Int64 uid = item.uniqueId();
      if (m_correspondance_functor){
        uid = m_correspondance_functor->getOldUniqueId(uid,iitem.index());
      }
      typename HashTableMapT<Int64,DataType>::Data* data = values_from_uid.lookup(uid);
      if (!data)
        throw FatalErrorException(A_FUNCINFO,
                                  String::format("Can not find item uid='{0}' reading variable '{1}'",
                                                 uid,var->fullName()));
      DataType value = data->value();
      var_value[iitem.index()] = value;
    }
  }
  else{
    Int32UniqueArray local_ids(buf_size);
    var->itemFamily()->itemsUniqueIdToLocalId(local_ids,unique_ids,false);
    for( Integer z=0; z<buf_size; ++z ){
      Integer lid = local_ids[z];
      if (lid==NULL_ITEM_LOCAL_ID)
        continue;
      if (lid>nb_var_value)
        throw FatalErrorException(A_FUNCINFO,String::format("Bad item index '{0}' max={1}",lid,nb_var_value));
      var_value[lid] = buffer[z];
    }
  }
  tm->info(4) << "End of read for variable '" << var->fullName() << "'";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ItemVariableInfo<VariableType,DataType>::
_readStandardArray(Array<DataType>& buffer,Int64Array& unique_ids,
                   const String& ids_hpath,hid_t file_id,
                   Hdf5Utils::StandardTypes& st)
{
  ARCANE_UNUSED(ids_hpath);

  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  ITraceMng* tm = vm->traceMng();
  IParallelMng* pm = vm->parallelMng();
  bool is_master = pm->isMasterIO();

  Hdf5Utils::StandardArrayT<DataType> values(file_id,path());

  if (is_master){
    if (!ids_hpath.null())
      values.setIdsPath(ids_hpath);
    values.readDim();
    Int64ConstArrayView dims(values.dimensions());
    Integer nb_dim = dims.size();
    if (nb_dim!=1)
      tm->fatal() << "Only one-dimension array are allowed "
                  << " dim=" << nb_dim << " var_name=" << var->fullName() << " path=" << path();
    Integer nb_item = arcaneCheckArraySize(dims[0]);
    tm->info(4) << "NB_ITEM: nb_item=" << nb_item;
    buffer.resize(nb_item);
    unique_ids.resize(nb_item);
  }
  values.parallelRead(pm,st,buffer,unique_ids);
#if 0
  {
    Integer index=0;
    Integer nb_item = buffer.size();
    for( Integer i=0; i<nb_item; ++i ){
      ++index;
      tm->info() << " VAR_VAL i=" << i << " value=" << buffer[i] << " uid=" << unique_ids[i];
      if (index>20)
        break;
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ItemVariableInfo<VariableType,DataType>::
_writeStandardArray(Array<DataType>& buffer,hid_t file_id,Hdf5Utils::StandardTypes& st)
{
  Hdf5Utils::StandardArrayT<DataType> values(file_id,path());
  //ITraceMng* tm = m_mesh->traceMng();
  //tm->info() << "WRITE STANDARD ARRAY N=" << buffer.size();
  values.write(st,buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ItemVariableInfo<VariableType,DataType>::
writeVariable(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st)
{
  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  IParallelMng* pm = vm->parallelMng();
  ITraceMng* tm = vm->traceMng();

  ItemGroup group = m_variable.itemGroup();
  ItemGroup enumerate_group = group.own();
  if (var->isPartial())
    enumerate_group = group;

  Integer nb_item = enumerate_group.size();

  UniqueArray<DataType> values(nb_item);
  Int64UniqueArray unique_ids(nb_item);
  tm->info(4) << "WRITE VARIABLE name=" << m_variable.name()
              << " is_partial=" << m_variable.variable()->isPartial();

  {
    Integer index = 0;
    ENUMERATE_ITEM(iitem,enumerate_group){
      if (iitem->isOwn()){
        values[index] = m_variable[iitem];
        unique_ids[index] = iitem->uniqueId();
        //tm->info() << "WRITE uid=" << iitem->uniqueId()
        //           << " value=" << m_variable[iitem];
        ++index;
      }
    }
    values.resize(index);
    unique_ids.resize(index);
  }
  // Pour l'instant la méthode parallèle créé un tampon
  // du nombre total d'éléments du tableau
  ///TODO a optimiser
  {
    Hdf5Utils::StandardArrayT<DataType> values_writer(hfile.id(),path());
    values_writer.parallelWrite(pm,st,values,unique_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType>
Hdf5ScalarVariableInfo<VariableType,DataType>::
Hdf5ScalarVariableInfo(IVariable* v)
: Hdf5VariableInfoBase()
, m_variable(v)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ScalarVariableInfo<VariableType,DataType>::
readVariable(Hdf5Utils::HFile& hfile,const String& filename,
             Hdf5Utils::StandardTypes& st,const String& ids_hpath,IData* data)
{
  ARCANE_UNUSED(ids_hpath);

  UniqueArray<DataType> buffer;
  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  ITraceMng* tm = vm->traceMng();
  IParallelMng* pm = vm->parallelMng();
  bool is_master = pm->isMasterIO();
  //Integer master_rank = pm->masterIORank();
  if (is_master){
    if (hfile.id()<0)
      hfile.openRead(filename);
  }

  Int64UniqueArray unique_ids;
  _readStandardArray(buffer,hfile.id(),st);
  
  Integer buf_size = buffer.size();
  if (!data)
    data = m_variable.variable()->data();
  IArrayDataT<DataType>* true_data = dynamic_cast<IArrayDataT<DataType>*>(data);
  if (!true_data)
    throw FatalErrorException("Can not convert IData to IArrayDataT");
  ArrayView<DataType> var_value = m_variable.asArray();
  for( Integer z=0; z<buf_size; ++z )
    var_value[z] = buffer[z];
  tm->info(4) << "End of read for variable '" << var->fullName() << "'";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ScalarVariableInfo<VariableType,DataType>::
_readStandardArray(Array<DataType>& buffer,hid_t file_id,Hdf5Utils::StandardTypes& st)
{
  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  ITraceMng* tm = vm->traceMng();
  IParallelMng* pm = vm->parallelMng();
  bool is_master = pm->isMasterIO();

  Hdf5Utils::StandardArrayT<DataType> values(file_id,path());

  if (is_master){
    values.readDim();
    Int64ConstArrayView dims(values.dimensions());
    Integer nb_dim = dims.size();
    if (nb_dim!=1)
      tm->fatal() << "Only one-dimension array are allowed "
                  << " dim=" << nb_dim << " var_name=" << var->fullName() << " path=" << path();
    Integer nb_item = arcaneCheckArraySize(dims[0]);
    tm->info(4) << "NB_ITEM: nb_item=" << nb_item;
    buffer.resize(nb_item);
  }
  values.read(st,buffer);
#if 0
  {
    Integer index=0;
    Integer nb_item = buffer.size();
    for( Integer i=0; i<nb_item; ++i ){
      ++index;
      tm->info() << " VAR_VAL i=" << i << " value=" << buffer[i];
      if (index>20)
        break;
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ScalarVariableInfo<VariableType,DataType>::
_writeStandardArray(Array<DataType>& buffer,hid_t file_id,Hdf5Utils::StandardTypes& st)
{
  Hdf5Utils::StandardArrayT<DataType> values(file_id,path());
  values.write(st,buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableType,typename DataType> void
Hdf5ScalarVariableInfo<VariableType,DataType>::
writeVariable(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st)
{
  IVariable* var = m_variable.variable();
  IVariableMng* vm = var->variableMng();
  IParallelMng* pm = vm->parallelMng();
  ITraceMng* tm = vm->traceMng();
  bool is_master = pm->isMasterIO();

  ConstArrayView<DataType> var_values = m_variable.asArray();
  Integer size = var_values.size();
  UniqueArray<DataType> values(size);
  for( Integer i=0; i<size; ++i )
    values[i] = var_values[i];

  // Comme il s'agit d'une variable scalaire, on considère que tous
  // les processeurs ont la même valeur. Donc seul le processeur
  // maitre écrit.
  if (is_master){
    //{
    Int64UniqueArray unique_ids;
    Hdf5Utils::StandardArrayT<DataType> values_writer(hfile.id(),path());
    tm->info(4) << "WRITE SCALAR VARIABLE name=" << m_variable.variable()->fullName();
    values_writer.write(st,values);
    //values_writer.parallelWrite(pm,st,values,unique_ids);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
