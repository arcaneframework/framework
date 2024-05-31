// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableScalar.cc                               (C) 2000-2024 */
/*                                                                           */
/* Variable scalaire sur un matériau du maillage.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialVariable.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Array2.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"
#include "arcane/materials/ItemMaterialVariableBaseT.H"
#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/internal/MeshMaterialVariablePrivate.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/internal/IVariableInternal.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/VariableRefArray.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/VariableArray.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/Timer.h"
#include "arcane/core/parallel/IStat.h"
#include "arcane/core/datatype/DataTypeTraits.h"
#include "arcane/core/datatype/DataStorageBuildInfo.h"
#include "arcane/core/VariableDataTypeTraits.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableScalarTraits<DataType>::
saveData(IMeshComponent* component,IData* data,
         Array<ContainerViewType>& cviews)
{
  ConstArrayView<ArrayView<DataType>> views = cviews;
  auto* true_data = dynamic_cast<ValueDataType*>(data);
  ARCANE_CHECK_POINTER(true_data);
  ContainerType& values = true_data->_internal()->_internalDeprecatedValue();
  ENUMERATE_COMPONENTCELL(icell,component){
    MatVarIndex mvi = icell._varIndex();
    values.add(views[mvi.arrayIndex()][mvi.valueIndex()]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableScalarTraits<DataType>::
copyTo(SmallSpan<const DataType> input, SmallSpan<const Int32> input_indexes,
       SmallSpan<DataType> output, SmallSpan<const Int32> output_indexes,
       const RunQueue& queue)
{
  // TODO: vérifier tailles des indexes identiques
  Integer nb_value = input_indexes.size();
  auto command = makeCommand(queue);
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, input.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, input_indexes.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output_indexes.data());
  command << RUNCOMMAND_LOOP1(iter, nb_value)
  {
    auto [i] = iter();
    output[output_indexes[i]] = input[input_indexes[i]];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableScalarTraits<DataType>::
resizeAndFillWithDefault(ValueDataType* data,ContainerType& container,Integer dim1_size)
{
 ARCANE_UNUSED(data);
 //TODO: faire une version de Array2 qui spécifie une valeur à donner
 // pour initialiser lors d'un resize() (comme pour Array::resize()).
 container.resize(dim1_size,DataType());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableScalarTraits<DataType>::
resizeWithReserve(PrivatePartType* var, Integer dim1_size, Real reserve_ratio)
{
  // Pour éviter de réallouer à chaque fois qu'il y a une augmentation du
  // nombre de mailles matériaux, alloue un petit peu plus que nécessaire.
  // Par défaut, on alloue 5% de plus.
  Int32 nb_add = static_cast<Int32>(dim1_size * reserve_ratio);
  var->_internalApi()->resizeWithReserve(dim1_size, nb_add);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ItemMaterialVariableScalar<DataType>::
ItemMaterialVariableScalar(const MaterialVariableBuildInfo& v,
                           PrivatePartType* global_var,
                           VariableRef* global_var_ref,MatVarSpace mvs)
: BaseClass(v,global_var,global_var_ref,mvs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de la variable pour un matériau à partir d'un tableau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  matvar[imatcell] = values[index];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values)
{
  // TODO: faire une version avec IMeshComponent
  Integer index = 0;
  ENUMERATE_COMPONENTITEM(MatCell,imatcell,mat){
    MatCell mc = *imatcell;
    MatVarIndex mvi = mc._varIndex();
    setValue(mvi,values[index]);
    ++index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de la variable pour un matériau à partir d'un tableau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  matvar[imatcell] = values[index];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,
              Int32ConstArrayView indexes)
{
  ConstArrayView<MatVarIndex> mat_indexes = mat->_internalApi()->variableIndexer()->matvarIndexes();
  Integer nb_index = indexes.size();
  for( Integer i=0; i<nb_index; ++i ){
    MatVarIndex mvi = mat_indexes[indexes[i]];
    setValue(mvi,values[i]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  values[index] = matvar[imatcell];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
fillToArray(IMeshMaterial* mat,ArrayView<DataType> values)
{
  Integer index=0;
  ENUMERATE_COMPONENTITEM(MatCell,imatcell,mat){
    MatCell mc = *imatcell;
    MatVarIndex mvi = mc._varIndex();
    values[index] = this->operator[](mvi);
    ++index;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  values[index] = matvar[imatcell];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes)
{
  ConstArrayView<MatVarIndex> mat_indexes = mat->_internalApi()->variableIndexer()->matvarIndexes();
  Integer nb_index = indexes.size();
  for( Integer i=0; i<nb_index; ++i ){
    MatVarIndex mvi = mat_indexes[indexes[i]];
    values[i] = this->operator[](mvi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs partielles avec la valeur \a value.
 */
template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
fillPartialValues(const DataType& value)
{
  // La variable d'indice 0 correspondant à la variable globale.
  // Il ne faut donc pas la prendre en compte.
  Integer nb_var = m_vars.size();
  for( Integer i=1; i<nb_var; ++i ){
    if (m_vars[i])
      m_vars[i]->fill(value);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Int32
ItemMaterialVariableScalar<DataType>::
dataTypeSize() const
{
  return (Int32)sizeof(DataType);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
_copyToBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,Span<std::byte> bytes) const
{
  // TODO: Vérifier que la taille est un multiple de sizeof(DataType) et que
  // l'alignement est correct.
  const Integer value_size = arcaneCheckArraySize(bytes.size() / sizeof(DataType));
  ArrayView<DataType> values(value_size,reinterpret_cast<DataType*>(bytes.data()));
  for( Integer z=0; z<value_size; ++z ){
    values[z] = this->operator[](matvar_indexes[z]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteArrayView bytes) const
{
  auto* ptr = reinterpret_cast<std::byte*>(bytes.data());
  return _copyToBufferLegacy(matvar_indexes,{ptr,bytes.size()});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
_copyFromBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,Span<const std::byte> bytes)
{
  // TODO: Vérifier que la taille est un multiple de sizeof(DataType) et que
  // l'alignement est correct.
  const Int32 value_size = CheckedConvert::toInt32(bytes.size() / sizeof(DataType));
  ConstArrayView<DataType> values(value_size,reinterpret_cast<const DataType*>(bytes.data()));
  for( Integer z=0; z<value_size; ++z ){
    setValue(matvar_indexes[z],values[z]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteConstArrayView bytes)
{
  auto* ptr = reinterpret_cast<const std::byte*>(bytes.data());
  return _copyFromBufferLegacy(matvar_indexes,{ptr,bytes.size()});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
synchronize()
{
  IParallelMng* pm = m_p->materialMng()->mesh()->parallelMng();
  Timer timer(pm->timerMng(),"MatTimer",Timer::TimerReal);
  Int64 message_size = 0;
  {
    Timer::Sentry ts(&timer);
    message_size = _synchronize2();
  }
  pm->stat()->add("MaterialSync",timer.lastActivationTime(),message_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Int64
ItemMaterialVariableScalar<DataType>::
_synchronize2()
{
  Int64 message_size = 0;
  Integer sync_version = m_p->materialMng()->synchronizeVariableVersion();
  // Seules les versions 6 et ultérieures sont disponibles pour les variables milieux.
  if (m_p->space()==MatVarSpace::Environment){
    if (sync_version<6)
      sync_version = 6;
  }
  if (sync_version>=6){
    MeshMaterialVariableSynchronizerList mmvsl(m_p->materialMng());
    mmvsl.add(this);
    mmvsl.apply();
    message_size = mmvsl.totalMessageSize();
  }
  else if (sync_version==5 || sync_version==4 || sync_version==3){
    _synchronizeV5();
  }
  else if (sync_version==2){
    _synchronizeV2();
  }
  else
    _synchronizeV1();
  return message_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
synchronize(MeshMaterialVariableSynchronizerList& sync_list)
{
  Integer sync_version = m_p->materialMng()->synchronizeVariableVersion();
  if (sync_version>=6){
    sync_list.add(this);
  }
  else
    this->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
_synchronizeV1()
{
  // Synchronisation:
  // Pour l'instant, algorithme non optimisée qui effectue plusieurs
  // synchros. On utilise la variable globale pour cela et il faut
  // donc la sauvegarder en début de fonction et la restorer en fin.
  // Le principe est simple: pour chaque materiau et milieu, on
  // recopie dans la variable globale la valeur partielle correspondante
  // et on fait une synchro de la variable globale. Il faut ensuite
  // recopier depuis la valeur globale dans la valeur partielle.
  // Cela signifie qu'on a autant de synchros que de matériaux et milieux

  // TODO: vérifier que la liste des matériaux est cohérente entre
  // les sous-domaines. Cela n'est pas nécessaire avec cet algo simplifié
  // mais le sera avec l'algo final. Pour ce test, il suffit de mettre
  // une valeur spéciale dans la variable globale pour chaque synchro
  // (par exemple MIN_FLOAT) et apres synchro de regarder pour chaque
  // maille dont la valeur est différente de MIN_FLOAT
  // si elle est bien dans notre matériau
  IMeshMaterialMng* material_mng = m_p->materialMng();
  IMesh* mesh = material_mng->mesh();
  //TODO: Utiliser autre type que cellFamily() pour permettre autre genre
  // d'élément.
  IItemFamily* family = mesh->cellFamily();
  IParallelMng* pm = mesh->parallelMng();
  if (!pm->isParallel())
    return;
  ItemGroup all_items = family->allItems();

  UniqueArray<DataType> saved_values;
  // Sauve les valeurs de la variable globale car on va les changer
  {
    ConstArrayView<DataType> var_values = m_global_variable->valueView();
    saved_values.resize(var_values.size());
    saved_values.copy(var_values);
  }

  ENUMERATE_ENV(ienv,material_mng){
    IMeshEnvironment* env = *ienv;
    ENUMERATE_MAT(imat,env){
      IMeshMaterial* mat = *imat;
      
      {
        ArrayView<DataType> var_values = m_global_variable->valueView();
        // Copie valeurs du matériau dans la variable globale puis synchro
        ENUMERATE_MATCELL(imatcell,mat){
          MatCell mc = *imatcell;
          Cell c = mc.globalCell();
          MatVarIndex mvi = mc._varIndex();
          var_values[c.localId()] = this->operator[](mvi);
        }
      }
      m_global_variable->synchronize();
      {
        ConstArrayView<DataType> var_values = m_global_variable->valueView();
        // Copie valeurs depuis la variable globale vers la partie materiau
        ENUMERATE_MATCELL(imatcell,mat){
          MatCell mc = *imatcell;
          Cell c = mc.globalCell();
          MatVarIndex mvi = mc._varIndex();
          setValue(mvi,var_values[c.localId()]);
        }
      }

    }

    // Effectue la même chose pour le milieu
    {
      ArrayView<DataType> var_values = m_global_variable->valueView();
      // Copie valeurs du milieu dans la variable globale puis synchro
      ENUMERATE_ENVCELL(ienvcell,env){
        EnvCell ec = *ienvcell;
        Cell c = ec.globalCell();
        MatVarIndex mvi = ec._varIndex();
        var_values[c.localId()] = this->operator[](mvi);
      }
    }
    m_global_variable->synchronize();
    {
      ConstArrayView<DataType> var_values = m_global_variable->valueView();
      // Copie valeurs depuis la variable globale vers la partie milieu
      ENUMERATE_ENVCELL(ienvcell,env){
        EnvCell ec = *ienvcell;
        Cell c = ec.globalCell();
        MatVarIndex mvi = ec._varIndex();
        setValue(mvi,var_values[c.localId()]);
      }
    }
  }

  // Restore les valeurs de la variable globale.
  {
    ArrayView<DataType> var_values = m_global_variable->valueView();
    var_values.copy(saved_values);
  }
  m_global_variable->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
_synchronizeV2()
{
  // Synchronisation:
  // Pour l'instant, algorithme non optimisée qui effectue plusieurs
  // synchros. On utilise la variable globale pour cela et il faut
  // donc la sauvegarder en début de fonction et la restorer en fin.
  // Le principe est simple: pour chaque materiau et milieu, on
  // recopie dans la variable globale la valeur partielle correspondante
  // et on fait une synchro de la variable globale. Il faut ensuite
  // recopier depuis la valeur globale dans la valeur partielle.
  // Cela signifie qu'on a autant de synchros que de matériaux et milieux

  // TODO: vérifier que la liste des matériaux est cohérente entre
  // les sous-domaines. Cela n'est pas nécessaire avec cet algo simplifié
  // mais le sera avec l'algo final. Pour ce test, il suffit de mettre
  // une valeur spéciale dans la variable globale pour chaque synchro
  // (par exemple MIN_FLOAT) et apres synchro de regarder pour chaque
  // maille dont la valeur est différente de MIN_FLOAT
  // si elle est bien dans notre matériau
  IMeshMaterialMng* material_mng = m_p->materialMng();
  IMesh* mesh = material_mng->mesh();
  //TODO: Utiliser autre type que cellFamily() pour permettre autre genre
  // d'élément.
  IItemFamily* family = mesh->cellFamily();
  IParallelMng* pm = mesh->parallelMng();
  if (!pm->isParallel())
    return;
  ItemGroup all_items = family->allItems();
  ITraceMng* tm = pm->traceMng();
  tm->info(4) << "MAT_SYNCHRONIZE_V2 name=" << this->name();
  IDataFactoryMng* df = m_global_variable->dataFactoryMng();

  DataStorageTypeInfo storage_type_info(VariableInfo::_internalGetStorageTypeInfo(m_global_variable->dataType(),2,0));
  DataStorageBuildInfo storage_build_info(tm);
  String storage_full_type = storage_type_info.fullName();

  Ref<IData> xdata(df->createSimpleDataRef(storage_full_type,storage_build_info));
  auto* data = dynamic_cast< IArray2DataT<DataType>* >(xdata.get());
  if (!data)
    ARCANE_FATAL("Bad type");

  ConstArrayView<MeshMaterialVariableIndexer*> indexers = material_mng->_internalApi()->variablesIndexer();
  Integer nb_indexer = indexers.size();
  data->_internal()->_internalDeprecatedValue().resize(family->maxLocalId(),nb_indexer+1);
  Array2View<DataType> values(data->view());

  // Recopie les valeurs partielles dans le tableau.
  for( MeshMaterialVariableIndexer* indexer : indexers ){
    ConstArrayView<MatVarIndex> matvar_indexes = indexer->matvarIndexes();
    ConstArrayView<Int32> local_ids = indexer->localIds();
    for( Integer j=0, n=matvar_indexes.size(); j<n; ++j ){
      MatVarIndex mvi = matvar_indexes[j];
      values[local_ids[j]][mvi.arrayIndex()] = this->operator[](mvi);
    }
  }
  family->allItemsSynchronizer()->synchronizeData(data);

  // Recopie du tableau synchronisé dans les valeurs partielles.
  for( MeshMaterialVariableIndexer* indexer : indexers ){
    ConstArrayView<MatVarIndex> matvar_indexes = indexer->matvarIndexes();
    ConstArrayView<Int32> local_ids = indexer->localIds();
    for( Integer j=0, n=matvar_indexes.size(); j<n; ++j ){
      MatVarIndex mvi = matvar_indexes[j];
      setValue(mvi,values[local_ids[j]][mvi.arrayIndex()]);
    }
  }

  m_global_variable->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
_synchronizeV5()
{
  // Version de la synchronisation qui envoie uniquement
  // les valeurs des matériaux et des milieux pour les mailles
  // partagées.
  // NOTE: Cette version nécessite que les matériaux soient correctement
  // synchronisés entre les sous-domaines.

  // Cette version est similaire à V4 dans son principe mais
  // fait les send/receive directement sans utiliser de sérialiser.
  IMeshMaterialMng* material_mng = m_p->materialMng();

  IMeshMaterialVariableSynchronizer* mmvs = material_mng->_internalApi()->allCellsMatEnvSynchronizer();
  IVariableSynchronizer* var_syncer = mmvs->variableSynchronizer();
  IParallelMng* pm = var_syncer->parallelMng();

  if (!pm->isParallel())
    return;

  mmvs->checkRecompute();

  //ItemGroup all_items = family->allItems();
  ITraceMng* tm = pm->traceMng();
  tm->info(4) << "MAT_SYNCHRONIZE_V5 name=" << this->name();
  //tm->info() << "STACK="<< platform::getStackTrace();
  const Integer data_type_size = (Integer)sizeof(DataType);
  {
    Int32ConstArrayView ranks = var_syncer->communicatingRanks();
    Integer nb_rank = ranks.size();
    std::vector< UniqueArray<DataType> > shared_values(nb_rank);
    std::vector< UniqueArray<DataType> > ghost_values(nb_rank);

    UniqueArray<Parallel::Request> requests;

    // Poste les receive.
    Int32UniqueArray recv_ranks(nb_rank);
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total = ghost_matcells.size();
      ghost_values[i].resize(total);
      Integer total_byte = CheckedConvert::multiply(total,data_type_size);
      ByteArrayView bytes(total_byte,(Byte*)(ghost_values[i].unguardedBasePointer()));
      requests.add(pm->recv(bytes,rank,false));
    }

    // Poste les send
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
      Integer total = shared_matcells.size();
      shared_values[i].resize(total);
      ArrayView<DataType> values(shared_values[i]);
      for( Integer z=0; z<total; ++z ){
        values[z] = this->operator[](shared_matcells[z]);
      }
      Integer total_byte = CheckedConvert::multiply(total,data_type_size);
      ByteArrayView bytes(total_byte,(Byte*)(values.unguardedBasePointer()));
      requests.add(pm->send(bytes,rank,false));
    }

    pm->waitAllRequests(requests);

    // Recopie les données recues dans les mailles fantomes.
    for( Integer i=0; i<nb_rank; ++i ){
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total = ghost_matcells.size();
      ConstArrayView<DataType> values(ghost_values[i].constView());
      for( Integer z=0; z<total; ++z ){
        setValue(ghost_matcells[z],values[z]);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
dumpValues(std::ostream& ostr,AllEnvCellVectorView view)
{
  ostr << "Dumping values for material variable name=" << this->name() << '\n';
  ENUMERATE_ALLENVCELL(iallenvcell,view){
    AllEnvCell all_env_cell = *iallenvcell;
    ostr << "Cell uid=" << ItemPrinter(all_env_cell.globalCell()) << " v=" << value(all_env_cell._varIndex()) << '\n';
    for( CellComponentCellEnumerator ienvcell(all_env_cell); ienvcell.hasNext(); ++ienvcell ){
      MatVarIndex evi = ienvcell._varIndex();
      ostr << "env_value=" << value(evi) << ", mvi=" << evi << '\n';
      for( CellComponentCellEnumerator imatcell(*ienvcell); imatcell.hasNext(); ++imatcell ){
        MatVarIndex mvi = imatcell._varIndex();
        ostr << "mat_value=" << value(mvi) << ", mvi=" << mvi << '\n';
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
dumpValues(std::ostream& ostr)
{
  ostr << "Dumping values for material variable name=" << this->name() << '\n';
  IItemFamily* family = m_global_variable->itemFamily();
  if (!family)
    return;
  IMeshMaterialMng* material_mng = m_p->materialMng();
  dumpValues(ostr,material_mng->view(family->allItems()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableScalar<DataType>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids)
{
  IItemFamily* family = m_global_variable->itemFamily();
  if (!family)
    return;
  ITraceMng* tm = family->traceMng();
  IMeshMaterialMng* mat_mng = m_p->materialMng();
  const Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  const eDataType data_type = DataTypeTraitsT<BasicType>::type();
  ItemVectorView ids_view(family->view(ids));
  bool has_mat = this->space()!=MatVarSpace::Environment;
  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    {
      Integer nb_val = 0;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
          EnvCell envcell = *ienvcell;
          ++nb_val; // 1 valeur pour le milieu
          if (has_mat)
            nb_val += envcell.nbMaterial(); // 1 valeur par matériau du milieu.
        }
      }
      tm->info() << "RESERVE: nb_value=" << 1 << " size=" << (nb_val*nb_count);
      sbuf->reserve(DT_Int64,1);  // Pour le nombre de valeurs.
      sbuf->reserveSpan(data_type,nb_val*nb_count);  // Pour le nombre de valeurs.
    }
    break;
  case ISerializer::ModePut:
    {
      UniqueArray<DataType> values;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
          values.add(value(ienvcell._varIndex()));
          if (has_mat){
            ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
              values.add(value(imatcell._varIndex()));
            }
          }
        }
      }
      Integer nb_value = values.size();
      ConstArrayView<BasicType> basic_values(nb_value*nb_count, reinterpret_cast<BasicType*>(values.data()));
      tm->info() << "PUT: nb_value=" << nb_value << " size=" << basic_values.size();
      sbuf->putInt64(nb_value);
      sbuf->putSpan(basic_values);
    }
    break;
  case ISerializer::ModeGet:
    {
      UniqueArray<BasicType> basic_values;
      Int64 nb_value = sbuf->getInt64();
      basic_values.resize(nb_value*nb_count);
      sbuf->getSpan(basic_values);
      Span<const DataType> data_values(reinterpret_cast<DataType*>(basic_values.data()),nb_value);
      Integer index = 0;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
          EnvCell envcell = *ienvcell;
          setValue(ienvcell._varIndex(),data_values[index]);
          ++index;
          if (has_mat){
            ENUMERATE_CELL_MATCELL(imatcell,envcell){
              setValue(imatcell._varIndex(),data_values[index]);
              ++index;
            }
          }
        }
      }
    }
    break;
  default:
    throw NotSupportedException(A_FUNCINFO,"Invalid serialize");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
MeshMaterialVariableScalar<ItemType,DataType>::
MeshMaterialVariableScalar(const MaterialVariableBuildInfo& v,PrivatePartType* global_var,
                           VariableRefType* global_var_ref,MatVarSpace mvs)
: ItemMaterialVariableScalar<DataType>(v,global_var,global_var_ref,mvs)
, m_true_global_variable_ref(global_var_ref) // Sera détruit par la classe de base
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
MeshMaterialVariableScalar<ItemType,DataType>::
~MeshMaterialVariableScalar()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ItemMaterialVariableBase< MaterialVariableScalarTraits<type> >;\
  template class ItemMaterialVariableScalar<type>;\
  template class MeshMaterialVariableScalar<Cell,type>;\
  template class MeshMaterialVariableCommonStaticImpl<MeshMaterialVariableScalar<Cell,type>>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(Real);
ARCANE_INSTANTIATE_MAT(Real2);
ARCANE_INSTANTIATE_MAT(Real3);
ARCANE_INSTANTIATE_MAT(Real2x2);
ARCANE_INSTANTIATE_MAT(Real3x3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
