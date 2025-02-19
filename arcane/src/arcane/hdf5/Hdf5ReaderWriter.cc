// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5ReaderWriter.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Lecture/Ecriture au format HDF5.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ArrayShape.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/StdNum.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/CheckpointService.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/VerifierService.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IData.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"

#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/internal/SerializeMessage.h"

#include "arcane/hdf5/Hdf5ReaderWriter.h"

#include "arcane/hdf5/Hdf5ReaderWriter_axl.h"

#include <array>
//#define ARCANE_TEST_HDF5MPI

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Hdf5Utils;

static herr_t _Hdf5ReaderWriterIterateMe(hid_t,const char*,void*);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
constexpr Int32 VARIABLE_INFO_SIZE = 10 + ArrayShape::MAX_NB_DIMENSION;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5ReaderWriter::
Hdf5ReaderWriter(ISubDomain* sd,const String& filename,
                 const String& sub_group_name,
                 Integer fileset_size, Integer currentIndex, Integer index_modulo,
                 eOpenMode open_mode,[[maybe_unused]] bool do_verif)
: TraceAccessor(sd->traceMng())
, m_parallel_mng(sd->parallelMng())
, m_open_mode(open_mode)
, m_filename(filename)
, m_sub_group_name(sub_group_name)
, m_is_initialized(false)
, m_io_timer(sd,"Hdf5Timer",Timer::TimerReal)
, m_is_parallel(false)
, m_my_rank(m_parallel_mng->commRank())
, m_send_rank(m_my_rank)
, m_last_recv_rank(m_my_rank)
, m_fileset_size(fileset_size)
, m_index_write(currentIndex)
, m_index_modulo(index_modulo)
{
  
  if (m_fileset_size!=1 && m_parallel_mng->isParallel()){
    m_is_parallel = true;
    Integer nb_rank = m_parallel_mng->commSize();
    if (m_fileset_size==0){
      m_send_rank = 0;
      m_last_recv_rank = nb_rank;
    }
    else{
      m_send_rank = (m_my_rank / m_fileset_size) * m_fileset_size;
      m_last_recv_rank = m_send_rank + m_fileset_size;
      if (m_last_recv_rank>nb_rank)
        m_last_recv_rank = nb_rank;
      --m_last_recv_rank;
    }
  }
  sd->traceMng()->info() << " INFOS PARALLEL: my_rank=" << m_my_rank
                         << " send_rank=" << m_send_rank
                         << " last_recv_rank=" << m_last_recv_rank
                         << " filename=" << filename;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
void Hdf5ReaderWriter::
initialize()
{
  if (m_is_initialized)
    return;
  m_is_initialized = true;
  HInit();
  info() << "INIT HDF5 READER/WRITER";
  {
    unsigned vmajor = 0;
    unsigned vminor = 0;
    unsigned vrel = 0;
    ::H5get_libversion(&vmajor,&vminor,&vrel);
    info() << "HDF5 version = " << vmajor << '.' << vminor << '.' << vrel;
  }
  info() << "SubGroup is '" << m_sub_group_name <<"'";
  if (m_open_mode==OpenModeRead){
    m_file_id.openRead(m_filename);
    m_sub_group_id.recursiveOpen(m_file_id,m_sub_group_name);
  }
  else{
    // Si ce n'est pas moi qui écrit, n'ouvre pas le fichier
    if (m_send_rank!=m_my_rank)
      return;
    if (m_open_mode==OpenModeTruncate){
      hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
#ifdef ARCANE_TEST_HDF5MPI
      void* arcane_comm = subDomain()->parallelMng()->getMPICommunicator();
      if (!arcane_comm)
        ARCANE_FATAL("No MPI environment available");
      MPI_Comm mpi_comm = *((MPI_Comm*)arcane_comm);
      MPI_Info mpi_info = MPI_INFO_NULL;
      //H5Pset_fapl_mpiposix(plist_id, mpi_comm, MPI_INFO_NULL); //mpi_info);
      H5Pset_fapl_mpio(plist_id, mpi_comm, MPI_INFO_NULL); //mpi_info);
      H5Pset_fclose_degree(plist_id,H5F_CLOSE_STRONG);
#endif
      int mdc_nelmts;
      size_t rdcc_nelmts;
      size_t rdcc_nbytes;
      double rdcc_w0;
      herr_t r = H5Pget_cache(plist_id,&mdc_nelmts,&rdcc_nelmts,&rdcc_nbytes,&rdcc_w0);
      info() << " CACHE SIZE r=" << r << " mdc=" << mdc_nelmts
             << " rdcc=" << rdcc_nelmts << " rdcc_bytes=" << rdcc_nbytes << " w0=" << rdcc_w0;
      mdc_nelmts *= 10;
      rdcc_nelmts *= 10;
      rdcc_nbytes = 10000000;
      r = H5Pset_cache(plist_id,mdc_nelmts,rdcc_nelmts,rdcc_nbytes,rdcc_w0);
      info() << " SET CACHE SIZE R1=" << r;
      //r = H5Pset_fapl_stdio(plist_id);
      //info() << " R2=" << r;
      hsize_t sieve_buf = (1024 << 12);
      r = H5Pset_sieve_buf_size(plist_id,sieve_buf);
      info() << " SIEVE_BUF=" << sieve_buf << " r=" << r;
      hsize_t small_block_size = 0;
      r = H5Pget_small_data_block_size(plist_id,&small_block_size);
      info() << " SMALL BLOCK SIZE=" << small_block_size;
      small_block_size <<= 10;
      r = H5Pset_small_data_block_size(plist_id,small_block_size);
      info() << " SET SMALL BLOCK SIZE s=" << small_block_size << " r=" << r;

      m_file_id.openTruncate(m_filename,plist_id);
    }
    else if (m_open_mode==OpenModeAppend){
      m_file_id.openAppend(m_filename);
    }
    if (m_sub_group_name!="/"){
      m_sub_group_id.checkDelete(m_file_id,m_sub_group_name);
      m_sub_group_id.recursiveCreate(m_file_id,m_sub_group_name);
    }
    else
      m_sub_group_id.open(m_file_id,m_sub_group_name);
  }
  if (m_file_id.isBad())
    ARCANE_THROW(ReaderWriterException,"Unable to open file '{0}'",m_filename);

  if (m_sub_group_id.isBad())
    ARCANE_THROW(ReaderWriterException,"HDF5 group '{0}' not found",m_sub_group_name);

  if (m_open_mode==OpenModeRead){
    int index = 0;
    //H5Giterate(m_sub_group_id.id(),"Variables",&index,_Hdf5ReaderWriterIterateMe,this);
    H5Giterate(m_file_id.id(),m_sub_group_name.localstr(),&index,_Hdf5ReaderWriterIterateMe,this);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5ReaderWriter::
~Hdf5ReaderWriter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_checkValid()
{
  if (m_is_initialized)
    return;
  fatal() << "Use of a Hdf5ReaderWriter instance not initialized";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Hdf5ReaderWriter::
_variableGroupName(IVariable* var)
{
  return var->fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture en parallèle.
 *
 * \warning En cours de test, pas utilisable.
 */
void Hdf5ReaderWriter::
_writeValParallel(IVariable* v,const ISerializedData* sdata)
{
  SerializeBuffer sb;
  sb.setMode(ISerializer::ModeReserve);
  sb.reserve(DT_Int32,1);       // Pour indiquer la fin des envois
  sb.reserve(v->fullName());
  sb.reserve(m_sub_group_name); //!< Nom du groupe.
  sb.reserve(DT_Int32,1);       // Pour indiquer le rand duquel le message provient
  sdata->serialize(&sb);
  sb.allocateBuffer();
  sb.setMode(ISerializer::ModePut);
  sb.putInt32(1);               // Indique qu'il s'agit d'un message non vide
  sb.put(v->fullName());
  sb.put(m_sub_group_name);     //!< Nom du groupe.
  sb.put(m_my_rank);
  sdata->serialize(&sb);
  m_parallel_mng->sendSerializer(&sb,m_send_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_directReadVal(IVariable* v,IData* data)
{
  _checkValid();
  info(4) << "DIRECT READ VAL v=" << v->name();
  _readVal(v,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_directWriteVal(IVariable* v,IData* data)
{
  _checkValid();
  Ref<ISerializedData> sdata(data->createSerializedDataRef(false));
  if (m_is_parallel && m_send_rank!=m_my_rank){
    _writeValParallel(v,sdata.get());
  }
  else{
    _writeVal(v->fullName(),m_sub_group_name,sdata.get());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static herr_t
_Hdf5ReaderWriterIterateMe(hid_t g,const char* mn,void* ptr)
{
  Hdf5ReaderWriter* rw = reinterpret_cast<Hdf5ReaderWriter*>(ptr);
  return rw->iterateMe(g,mn);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t Hdf5ReaderWriter::
iterateMe(hid_t group_id,const char* member_name)
{
  ARCANE_UNUSED(group_id);
  m_variables_name.add(StringView(member_name));
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_writeVal(const String& var_group_name,
          const String& sub_group_name,
          const ISerializedData* sdata,
          const Int32 from_rank)
{
  const bool hits_modulo=(m_index_modulo!=0) && (m_index_write!=0) && ((m_index_write%m_index_modulo)==0);
  Timer::Sentry ts(&m_io_timer);

  info(4) << " SDATA name=" << var_group_name << " nb_element=" << sdata->nbElement()
          << " dim=" << sdata->nbDimension() << " datatype=" << sdata->baseDataType()
          << " nb_basic_element=" << sdata->nbBaseElement()
          << " is_multi=" << sdata->isMultiSize()
          << " dimensions_size=" << sdata->extents().size()
          << " memory_size=" << sdata->memorySize()
          << " bytes_size=" << sdata->constBytes().size()
          << " shape=" << sdata->shape().dimensions();

  Integer nb_dimension = sdata->nbDimension();
  Int64ConstArrayView dimensions = sdata->extents();

  hid_t save_typeid = m_types.saveType(sdata->baseDataType());
  hid_t trueid = m_types.nativeType(sdata->baseDataType());
  const void* ptr = sdata->constBytes().data();
  Int64 nb_base_element = sdata->nbBaseElement();

  HGroup var_base_group;
  var_base_group.recursiveCreate(m_file_id,sub_group_name);

  // Création du groupe contenant les informations de la variable
  HGroup group_id;
  group_id.recursiveCreate(var_base_group,var_group_name);
  if (group_id.isBad())
    ARCANE_THROW(ReaderWriterException,"HDF5 group '{0}' not found",var_group_name);

  Int64 nb_element = sdata->nbElement();
  bool is_multi_size = sdata->isMultiSize();
  Int64 dim2_size = 0;
  Int64 dim1_size = 0;
  if (nb_dimension==2 && !is_multi_size){
    dim1_size = dimensions[0];
    dim2_size = dimensions[1];
  }
  Integer dimension_array_size = dimensions.size();

  // Sauve les informations concernant les tailles et dimensions de la variable
  {
    hsize_t att_dims[1];
    att_dims[0] = VARIABLE_INFO_SIZE;
    HSpace space_id;
    space_id.createSimple(1,att_dims);
    std::array<Int64,VARIABLE_INFO_SIZE> dim_val_buf;
    SmallSpan<Int64> dim_val(dim_val_buf);
    dim_val.fill(0);

    dim_val[0] = nb_dimension;
    dim_val[1] = dim1_size;
    dim_val[2] = dim2_size;
    dim_val[3] = nb_element;
    dim_val[4] = nb_base_element;
    dim_val[5] = dimension_array_size;
    dim_val[6] = is_multi_size ? 1 : 0;
    dim_val[7] = sdata->baseDataType();
    dim_val[8] = sdata->memorySize();
    {
      ArrayShape shape = sdata->shape();
      Int32 shape_nb_dim = shape.nbDimension();
      auto shape_dims = shape.dimensions();
      dim_val[9] = shape_nb_dim;
      for (Integer i=0; i<shape_nb_dim; ++i )
        dim_val[10+i] = shape_dims[i];
    }
    HAttribute att_id;
    if (m_is_parallel && hits_modulo && (from_rank!=0))
      att_id.remove(group_id,"Dims");
    att_id.create(group_id,"Dims",m_types.saveType(dim1_size),space_id);
    herr_t herr = att_id.write(m_types.nativeType(dim2_size),dim_val.data());
    if (herr<0)
      ARCANE_THROW(ReaderWriterException,"Wrong dimensions written for variable '{0}'",var_group_name);
  }

  // Si la variable est de type tableau à deux dimensions, sauve les
  // tailles de la deuxième dimension par élément.
  if (dimension_array_size!=0){
    hsize_t att_dims[1];
    att_dims[0] = dimension_array_size;
    HSpace space_id;
    HDataset array_id;
    space_id.createSimple(1,att_dims);
    array_id.recursiveCreate(group_id,"Dim2",m_types.saveType(dim1_size),space_id,H5P_DEFAULT);
    herr_t herr = array_id.write(m_types.nativeType(dim1_size),dimensions.data());
    if (herr<0)
      ARCANE_THROW(ReaderWriterException,"Wrong dimensions written for variable '{0}'",var_group_name);
  }

  // Maintenant, sauve les valeurs si necessaire
  if (nb_base_element!=0 && ptr!=nullptr){
    debug(Trace::High) << "Variable " << var_group_name << " begin dumped (nb_base_element=" << nb_base_element << ").";
    hsize_t dims[1];
    dims[0] = nb_base_element;
    HSpace space_id;
    space_id.createSimple(1,dims);
    if (space_id.isBad())
      ARCANE_THROW(ReaderWriterException,"Wrong dataspace for variable '{0}'",var_group_name);
    
    HDataset dataset_id;
    hid_t plist_id = H5P_DEFAULT;

#if 0
    if (nb_element>=10000){
      plist_id = H5Pcreate(H5P_DATASET_CREATE);
      hsize_t chunk_dim[1];
      chunk_dim[0] = (4096 << 1);
      herr_t r = H5Pset_chunk(plist_id,1,chunk_dim);
      info() << " SET CHUNK FOR " << var_group_name << " s=" << nb_element;
    }
#endif
    dataset_id.recursiveCreate(group_id,"Values",save_typeid,space_id,plist_id);
    if (dataset_id.isBad())
      ARCANE_THROW(ReaderWriterException,"Wrong dataset for variable '{0}'",var_group_name);

    herr_t herr = dataset_id.write(trueid,ptr);
    if (herr<0)
      ARCANE_THROW(ReaderWriterException,"Wrong dataset written for variable '{0}'",var_group_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> Hdf5ReaderWriter::
_readDim2(IVariable* var)
{
  const int max_dim = 256; // Nombre maxi de dimensions des tableaux HDF
  String vname = _variableGroupName(var);
  info(4) << " READ DIM name=" << vname;
  Int64 dimension_array_size = 0;
  Int64 nb_element = 0;
  Integer nb_dimension = -1;
  // Regarde si le nom correspondant est dans la liste des variables.
  // S'il n'y est pas, cela signifie que le tableau n'a pas été sauvé et
  // donc que ses dimensions sont nulles.
  {
    bool is_found = false;
    for( StringList::Enumerator i(m_variables_name); ++i; )
      if (*i==vname){
        is_found = true;
        break;
      }
    if (!is_found)
      ARCANE_THROW(ReaderWriterException,"No HDF5 group named '{0} exists",vname);
  }

  // Récupère le groupe contenant les informations de la variable
  HGroup group_id;
  //group_id.open(m_variable_group_id,vname);
  group_id.open(m_sub_group_id,vname);
  if (group_id.isBad())
    ARCANE_THROW(ReaderWriterException,"HDF5 group '{0}' not found",vname);

  bool is_multi_size = false;
  eDataType data_type = DT_Unknown;
  Int64 memory_size = 0;
  Int64 nb_base_element = 0;
  Int64 dim1_size = 0;
  Int64 dim2_size = 0;
  UniqueArray<Int64> dims;
  ArrayShape data_shape;

  // Récupère les informations concernant les tailles et dimensions de la variable
  {
    HAttribute att_id;
    att_id.open(group_id,"Dims");
    HSpace space_id = att_id.getSpace();
		
    // On attend une seule dimension, et le nombre d'eléments de
    // l'attribut (hdf_dims[0]) doit être égal à 1 ou 2.
    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    H5Sget_simple_extent_dims(space_id.id(),hdf_dims,max_dims);

    if (hdf_dims[0]!=VARIABLE_INFO_SIZE)
      ARCANE_THROW(ReaderWriterException,"Wrong dimensions for variable '{0}' (found={1} expected={2})",
                   vname, hdf_dims[0], VARIABLE_INFO_SIZE);

    std::array<Int64,VARIABLE_INFO_SIZE> dim_val_buf;
    att_id.read(m_types.nativeType(Int64()),dim_val_buf.data());

    SmallSpan<const Int64> dim_val(dim_val_buf);

    nb_dimension = CheckedConvert::toInteger(dim_val[0]);
    dim1_size = dim_val[1];
    dim2_size = dim_val[2];
    nb_element = dim_val[3];
    nb_base_element = dim_val[4];
    dimension_array_size = dim_val[5];
    is_multi_size = dim_val[6]!=0;
    data_type = (eDataType)dim_val[7];
    memory_size = dim_val[8];
    Int32 shape_nb_dim =  CheckedConvert::toInt32(dim_val[9]);
    data_shape.setNbDimension(shape_nb_dim);
    for (Integer i=0; i<shape_nb_dim; ++i )
      data_shape.setDimension(i,CheckedConvert::toInt32(dim_val[10+i]));
  }

  info(4) << " READ DIM name=" << vname
          << " nb_dim=" << nb_dimension << " dim1_size=" << dim1_size
          << " dim2_size=" << dim2_size << " nb_element=" << nb_element
          << " dimension_size=" << dimension_array_size
          << " is_multi_size=" << is_multi_size
          << " data_type" << data_type
          << " shape=" << data_shape.dimensions();

  if (dimension_array_size>0){
    HDataset array_id;
    array_id.open(group_id,"Dim2");
    if (array_id.isBad())
      ARCANE_THROW(ReaderWriterException,"Wrong dataset for variable '{0}'",vname);

    HSpace space_id = array_id.getSpace();
    if (space_id.isBad())
      ARCANE_THROW(ReaderWriterException,"Wrong dataspace for variable '{0}'",vname);

    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    H5Sget_simple_extent_dims(space_id.id(),hdf_dims,max_dims);
    // Vérifie que le nombre d'éléments du dataset est bien égal à celui
    // attendu.
    if ((Int64)hdf_dims[0]!=dimension_array_size){
      ARCANE_THROW(ReaderWriterException,"Wrong number of elements in 'Dim2' for variable '{0}' (found={1} expected={2})",
                   vname, hdf_dims[0], dimension_array_size);
                   
    }
    dim2_size = 0;
    dims.resize(dimension_array_size);
    herr_t herr = array_id.read(m_types.nativeType(Int64()),dims.data());
    if (herr<0)
      ARCANE_THROW(ReaderWriterException,"Wrong dataset read for variable '{0}'",vname);
  }
  Ref<ISerializedData> sdata = arcaneCreateSerializedDataRef(data_type,memory_size,nb_dimension,nb_element,
                                                             nb_base_element,is_multi_size,dims,data_shape);
  return sdata;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
write(IVariable* v,IData* data)
{
  _directWriteVal(v,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_readVal(IVariable* v,IData* data)
{
  String var_group_name = _variableGroupName(v);
  info(4) << " TRY TO READ var_group=" << var_group_name;
  Ref<ISerializedData> sd(_readDim2(v));
  Int64 storage_size = sd->memorySize();
  info(4) << " READ DATA n=" << storage_size;
  data->allocateBufferForSerializedData(sd.get());
  if (storage_size!=0){
    // Récupère le groupe contenant les informations de la variable
    HGroup group_id;
    //group_id.open(m_variable_group_id,var_group_name);
    group_id.open(m_sub_group_id,var_group_name);
    if (group_id.isBad())
      ARCANE_THROW(ReaderWriterException,"No HDF5 group with name '{0}' exists",var_group_name);
    HDataset dataset_id;
    dataset_id.open(group_id,"Values");
    if (dataset_id.isBad())
      ARCANE_THROW(ReaderWriterException,"Wrong dataset for variable '{0}'",var_group_name);
    void* ptr = sd->writableBytes().data();
    info() << "READ Variable " << var_group_name << " ptr=" << ptr;;
    hid_t trueid = m_types.nativeType(sd->baseDataType());
    dataset_id.read(trueid,ptr);
  }
  data->assignSerializedData(sd.get());
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
read(IVariable* var,IData* data)
{
  _directReadVal(var,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
setMetaData(const String& meta_data)
{
  if (m_is_parallel){
    IParallelMng* pm = m_parallel_mng;
    //Integer nb_rank = pm->commSize();
    if (m_send_rank!=m_my_rank){
      // Envoie le groupe et les meta donnees
      SerializeBuffer sb;
      sb.setMode(ISerializer::ModeReserve);
      sb.reserve(m_sub_group_name);
      sb.reserve(meta_data);
      sb.allocateBuffer();
      sb.setMode(ISerializer::ModePut);
      sb.put(m_sub_group_name);
      sb.put(meta_data);
      m_parallel_mng->sendSerializer(&sb,m_send_rank);
    }
    else{
      _setMetaData(meta_data,m_sub_group_name);
      for( Integer i=m_send_rank+1; i<=m_last_recv_rank; ++i ){
        SerializeBuffer sb;
        pm->recvSerializer(&sb,i);
        sb.setMode(ISerializer::ModeGet);
        String remote_group_name;
        String remote_meta_data;
        sb.get(remote_group_name);
        sb.get(remote_meta_data);
        _setMetaData(remote_meta_data,remote_group_name);
      }
    }
  }
  else
    _setMetaData(meta_data,m_sub_group_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_setMetaData(const String& meta_data,const String& sub_group_name)
{
  const bool hits_modulo=(m_index_modulo!=0) && (m_index_write!=0) && ((m_index_write%m_index_modulo)==0);
  HGroup base_group;
  if (hits_modulo)
    base_group.recursiveOpen(m_file_id,sub_group_name);
  else
    base_group.recursiveCreate(m_file_id,sub_group_name);
  
  ByteConstArrayView meta_data_utf8 = meta_data.utf8();
  const Byte* _meta_data = meta_data_utf8.data();
  hsize_t dims[1];
  dims[0] = meta_data_utf8.size() + 1;
  
  HSpace space_id;
  space_id.createSimple(1,dims);
  if (space_id.isBad())
    throw ReaderWriterException(A_FUNCINFO,"Wrong space for meta-data ('MetaData')");

  HDataset dataset_id;
  if (hits_modulo)
    dataset_id.recursiveCreate(base_group,"MetaData", m_types.nativeType(Byte()), space_id, H5P_DEFAULT);
  else
    dataset_id.create(base_group,"MetaData", m_types.nativeType(Byte()), space_id, H5P_DEFAULT);
  if (dataset_id.isBad())
    throw ReaderWriterException(A_FUNCINFO,"Wrong dataset for meta-data ('MetaData')");

  herr_t herr = dataset_id.write(m_types.nativeType(Byte()),_meta_data);
  if (herr<0)
    throw ReaderWriterException(A_FUNCINFO,"Unable to write meta-data ('MetaData')");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Hdf5ReaderWriter::
metaData()
{
  HDataset dataset_id;
  dataset_id.open(m_sub_group_id,"MetaData");
  if (dataset_id.isBad()){
    throw ReaderWriterException(A_FUNCINFO,"Wrong dataset for meta-data ('MetaData')");
  }
  HSpace space_id = dataset_id.getSpace();
  if (space_id.isBad()){
    throw ReaderWriterException(A_FUNCINFO,"Wrong space for meta-data ('MetaData')");
  }
  const int max_dim = 256;
  hsize_t hdf_dims[max_dim];
  hsize_t max_dims[max_dim];
  H5Sget_simple_extent_dims(space_id.id(),hdf_dims,max_dims);
  if (hdf_dims[0]<=0)
    throw ReaderWriterException(A_FUNCINFO,"Wrong number of elements for meta-data ('MetaData')");
  Integer nb_byte = static_cast<Integer>(hdf_dims[0]);
  ByteUniqueArray uchars(nb_byte);
  dataset_id.read(m_types.nativeType(Byte()),uchars.data());
  String s(uchars);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
endWrite()
{
  if (m_is_parallel){
    if (m_my_rank==m_send_rank){
      _receiveRemoteVariables();
    }
    else{
      // Envoie un message de fin
      SerializeBuffer sb;
      sb.setMode(ISerializer::ModeReserve);
      sb.reserve(DT_Int32,1); // Pour indiquer la fin des envoies
      sb.allocateBuffer();
      sb.setMode(ISerializer::ModePut);
      sb.putInt32(0); // Indique qu'il s'agit d'un message de fin
      m_parallel_mng->sendSerializer(&sb,m_send_rank);
    }
  }
  {
    info() << " Hdf5Timer: nb_activated=" << m_io_timer.nbActivated()
           << " time=" << m_io_timer.totalTime();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_receiveRemoteVariables()
{
  IParallelMng* pm = m_parallel_mng;
  Integer nb_remaining = m_last_recv_rank - m_send_rank;
  info() << "NB REMAINING = " << nb_remaining;
  Ref<ISerializeMessageList> m_messages(pm->createSerializeMessageListRef());
  while(nb_remaining>0){
    ScopedPtrT<ISerializeMessage> sm(new SerializeMessage(m_my_rank,NULL_SUB_DOMAIN_ID,ISerializeMessage::MT_Recv));
    m_messages->addMessage(sm.get());
    m_messages->processPendingMessages();
    m_messages->waitMessages(Parallel::WaitAll);
    ISerializer* sb = sm->serializer();
    sb->setMode(ISerializer::ModeGet);
    Int32 id = sb->getInt32();
    if (id==0)
      --nb_remaining;
    else
      _writeRemoteVariable(sb);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5ReaderWriter::
_writeRemoteVariable(ISerializer* sb)
{
  String var_name;
  sb->get(var_name);
  String group_name;
  sb->get(group_name);
  Int32 rank = sb->getInt32();
  //warning()<<"[\33[46;30m_writeRemoteVariable\33[m] rank="<<rank;
  Ref<ISerializedData> sdata = arcaneCreateEmptySerializedDataRef();
  sb->setReadMode(ISerializer::ReadReplace);
  sdata->serialize(sb);
  _writeVal(var_name,group_name,sdata.get(),rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Protection/reprise au format ArcaneHdf5.
 */
class ArcaneHdf5CheckpointService2
: public ArcaneHdf5ReaderWriterObject
{
 public:
  ArcaneHdf5CheckpointService2(const ServiceBuildInfo& sbi)
  : ArcaneHdf5ReaderWriterObject(sbi),
    m_write_index(0),
    m_writer(nullptr),
    m_reader(nullptr),
    m_fileset_size(1),
    m_index_modulo(0){}
  
  virtual IDataWriter* dataWriter() { return m_writer; }
  virtual IDataReader* dataReader() { return m_reader; }

  virtual void notifyBeginWrite();
  virtual void notifyEndWrite();
  virtual void notifyBeginRead();
  virtual void notifyEndRead();
  virtual void close() {}
  virtual String readerServiceName() const { return "ArcaneHdf5CheckpointReader2"; }

 private:

  Integer m_write_index;
  Hdf5ReaderWriter* m_writer;
  Hdf5ReaderWriter* m_reader;
  Integer m_fileset_size;
  Integer m_index_modulo;

 private:

  String _defaultFileName()
  {
    info() << "USE DEFAULT FILE NAME";
    IParallelMng* pm = subDomain()->parallelMng();
    Integer rank = pm->commRank();
    StringBuilder buf;

    // Ajoute si besoin le numero du processeur
    if (pm->isParallel()){
      Integer file_id = rank;
      if (m_fileset_size!=0)
        file_id = (rank / m_fileset_size) * m_fileset_size;
      buf = "arcanedump.";
      buf += file_id;
    }
    else{
      buf = "arcanedump";
    }

    // Ajoute si besoin le numero du replica
    IParallelReplication* pr = subDomain()->parallelMng()->replication();
    if (pr->hasReplication()){
      buf += "_r";
      buf += pr->replicationRank();
    }

    buf += ".h5";
    return buf.toString();
  }
  
  Directory _defaultDirectory(){
    return Directory(baseDirectoryName());
  }
  void _parseMetaData(String meta_data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5CheckpointService2::
_parseMetaData(String meta_data)
{
  IIOMng* io_mng = subDomain()->ioMng();
  ScopedPtrT<IXmlDocumentHolder> xml_doc(io_mng->parseXmlBuffer(meta_data.utf8(),"MetaData"));
  XmlNode root = xml_doc->documentNode().documentElement();
  Integer version = root.attr("version").valueAsInteger();
  if (version!=1){
    throw ReaderWriterException(A_FUNCINFO,"Bad version (expected 1)");
  }
  {
    Integer fileset_size = root.child("fileset-size").valueAsInteger();
    if (fileset_size<0) fileset_size = 0;
    m_fileset_size = fileset_size;
  }
  {
    Integer index_modulo = root.child("index-modulo").valueAsInteger();
    if (index_modulo<0) index_modulo = 0;
    m_index_modulo=index_modulo;
  }
  info() << " FileSet size=" << m_fileset_size;
  info() << " Index modulo=" << m_index_modulo;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5CheckpointService2::
notifyBeginRead()
{
  String meta_data = readerMetaData();
  _parseMetaData(meta_data);

  info() << " GET META DATA READER " << readerMetaData()
         << " filename=" << fileName();

  if (fileName().null()){
    Directory dump_dir(_defaultDirectory());
    setFileName(dump_dir.file(_defaultFileName()));
  }
  info() << " READ CHECKPOINT FILENAME = " << fileName();
  StringBuilder sub_group;
  sub_group = "SubDomain";
  sub_group += subDomain()->subDomainId();
  sub_group += "/Index";
  
  Integer index = currentIndex();
  if (m_index_modulo!=0)
    index %= m_index_modulo;
  sub_group += index;
  
  m_reader = new Hdf5ReaderWriter(subDomain(),
                                  fileName(),
                                  sub_group.toString(),
                                  0,
                                  currentIndex(),
                                  m_index_modulo,
                                  Hdf5ReaderWriter::OpenModeRead);
  m_reader->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5CheckpointService2::
notifyEndRead()
{
  delete m_reader;
  m_reader = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5CheckpointService2::
notifyBeginWrite()
{
  if (options()){
    // Récupération du nombre de fichiers par groupe
    m_fileset_size = options()->filesetSize();
    // Récupération du nombre d'indexes au maximum par fichiers
    m_index_modulo = options()->indexModulo();
  }

  if (fileName().null()){
    Directory dump_dir(_defaultDirectory());
    setFileName(dump_dir.file(_defaultFileName()));
  }
  Hdf5ReaderWriter::eOpenMode open_mode = Hdf5ReaderWriter::OpenModeAppend;
  Integer write_index = checkpointTimes().size();
  --write_index;
  
  if (write_index==0)
    open_mode = Hdf5ReaderWriter::OpenModeTruncate;

  // Test de l'option m_index_modulo pour savoir la profondeur du modulo
  if (m_index_modulo!=0)
    write_index%=m_index_modulo;

  StringBuilder sub_group;
  sub_group = "SubDomain";
  sub_group += subDomain()->parallelMng()->commRank();
  sub_group += "/Index";
  sub_group += write_index;
  
  m_writer = new Hdf5ReaderWriter(subDomain(),
                                  fileName(),
                                  sub_group,
                                  m_fileset_size,
                                  checkpointTimes().size()-1,
                                  m_index_modulo,
                                  open_mode);
  m_writer->initialize();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5CheckpointService2::
notifyEndWrite()
{
  OStringStream ostr;
  ostr() << "<infos version='1'>\n";
  ostr() << " <fileset-size>" << m_fileset_size << "</fileset-size>\n";
  ostr() << " <index-modulo>" << m_index_modulo << "</index-modulo>\n";
  ostr() << "</infos>\n";
  setReaderMetaData(ostr.str());
  ++m_write_index;
  delete m_writer;
  m_writer = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ArcaneHdf5CheckpointService2,
                        ServiceProperty("ArcaneHdf5CheckpointReader2",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICheckpointReader));

ARCANE_REGISTER_SERVICE(ArcaneHdf5CheckpointService2,
                        ServiceProperty("ArcaneHdf5CheckpointWriter2",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICheckpointWriter));

ARCANE_REGISTER_SERVICE_HDF5READERWRITER(ArcaneHdf5Checkpoint2,
                                         ArcaneHdf5CheckpointService2);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
