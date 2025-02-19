// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5MpiReaderWriter.cc                                      (C) 2000-2025 */
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

#include "arcane/core/Item.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataWriter.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/StdNum.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/CheckpointService.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/VerifierService.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IData.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/ISerializeMessageList.h"

#include "arcane/hdf5/Hdf5MpiReaderWriter.h"

#include "arcane/hdf5/Hdf5MpiReaderWriter_axl.h"

#include "arcane_packages.h"

#ifdef ARCANE_HAS_PACKAGE_MPI
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#include <mpi.h>

//#define ARCANE_TEST_HDF5MPI

// Pour l'instant (1.8.0 beta 2), cela ne fonctionne pas sur tera 10
// #define ARCANE_TEST_HDF5DIRECT

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Hdf5Utils;

static herr_t _Hdf5MpiReaderWriterIterateMe(hid_t,const char*,void*);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5MpiReaderWriter::
Hdf5MpiReaderWriter(ISubDomain* sd,const String& filename,
                 const String& sub_group_name,Integer fileset_size,
                 eOpenMode open_mode,bool do_verif)
: TraceAccessor(sd->traceMng())
, m_sub_domain(sd)
, m_parallel_mng(sd->parallelMng())
, m_open_mode(open_mode)
, m_filename(filename)
, m_sub_group_name(sub_group_name)
, m_is_initialized(false)
, m_io_timer(sd,"Hdf5TimerHd",Timer::TimerReal)
, m_write_timer(sd,"Hdf5TimerWrite",Timer::TimerReal)
, m_is_parallel(false)
, m_my_rank(m_parallel_mng->commRank())
, m_send_rank(m_my_rank)
, m_last_recv_rank(m_my_rank)
, m_fileset_size(fileset_size)
{
  ARCANE_UNUSED(do_verif);
  if (m_fileset_size!=1 && m_parallel_mng->isParallel()){
    m_is_parallel = true;
    Integer nb_rank = m_parallel_mng->commSize();
    if (m_fileset_size==0){
      m_send_rank = 0;
      m_last_recv_rank = nb_rank;
      --m_last_recv_rank;
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
                         << " filename=" << filename
                         << " fileset_size=" << m_fileset_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
initialize()
{
  if (m_is_initialized)
    return;

  m_is_initialized = true;

  const char* func_name = "Hdf5MpiReaderWriter::initialize()";

  HInit();

  if (m_open_mode==OpenModeRead){
    m_file_id.openRead(m_filename);
    m_sub_group_id.recursiveOpen(m_file_id,m_sub_group_name);
    //m_variable_group_id.open(m_sub_group_id,"Variables");
  }
  else{
    void* arcane_comm = m_sub_domain->parallelMng()->getMPICommunicator();
    if (!arcane_comm)
      throw FatalErrorException("No MPI environment available");
    MPI_Comm mpi_comm = *((MPI_Comm*)arcane_comm);
    Integer nb_rank = m_parallel_mng->commSize();
    if (m_fileset_size>1){
      UniqueArray<int> senders;
      for( Integer i=0; i<nb_rank; ++i ){
        Integer modulo = i % m_fileset_size;
        if (modulo==0){
          info() << " ADD SENDER n=" << i;
          senders.add(i);
        }
      }
      MPI_Group all_group;
      if (MPI_Comm_group(mpi_comm,&all_group)!=MPI_SUCCESS)
        fatal() << "Error in MPI_Comm_group";
      MPI_Group writer_group;
      if (MPI_Group_incl(all_group,senders.size(),senders.data(),&writer_group)!=MPI_SUCCESS)
        fatal() << "Error in MPI_Group_incl";
      if (MPI_Comm_create(mpi_comm,writer_group,&mpi_comm)!=MPI_SUCCESS)
        fatal() << "Error in MPI_Comm_create";
    }

    // Si ce n'est pas moi qui écrit, n'ouvre pas le fichier
    if (m_send_rank!=m_my_rank)
      return;
    if (m_open_mode==OpenModeTruncate || m_open_mode==OpenModeAppend){
      hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
      //bool use_gpfs = false;
      info() << " USE MPI-POSIX";
      //H5Pset_fapl_mpiposix(plist_id, mpi_comm, true);
#ifdef H5_HAVE_PARALLEL
      H5Pset_fapl_mpio(plist_id, mpi_comm, MPI_INFO_NULL); //mpi_info);
#endif

#ifdef ARCANE_TEST_HDF5DIRECT
#  ifdef H5_HAVE_DIRECT
      info() << " HAVE DIRECT DRIVER";
      H5Pset_fapl_direct(plist_id,4096,512,16*1024*1024);
#  endif
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
      //hsize_t block_size = 0;
      //block_size = H5Pget_buffer(plist_id,0,0);
      //info() << " BLOCK SIZE s=" << block_size;
      //block_size = 10000000;
      //herr_t r = H5Pset_buffer(plist_id,block_size,0,0);
      //info() << " BLOCK SIZE r=" << r << " s=" << block_size;
      //if (m_parallel_mng->commRank()==0){
      //herr_t r = H5Pset_fapl_core(plist_id,1000000,1);
      //else
      //herr_t r = H5Pset_fapl_core(plist_id,1000000,0);
      //m_file_id.openTruncate("toto",plist_id);
      //}
      //else
      if (m_open_mode==OpenModeTruncate){
        info() << " BEGIN OPEN TRUNCATE";
        m_file_id.openTruncate(m_filename,plist_id);
        info() << " END OPEN TRUNCATE";
      }
      else if (m_open_mode==OpenModeAppend){
        info() << " BEGIN OPEN ADD";
        m_file_id.openAppend(m_filename,plist_id);
        info() << " END OPEN ADD";
      }
    }
    if (m_sub_group_name!="/"){
      info() << " CHECK CREATE GROUP name=" << m_sub_group_name;
      //m_sub_group_id.checkDelete(m_file_id,m_sub_group_name);
      m_sub_group_id.recursiveCreate(m_file_id,m_sub_group_name);
      info() << " END CHECK CREATE GROUP name=" << m_sub_group_name;
    }
    else
      m_sub_group_id.open(m_file_id,m_sub_group_name);
    m_variable_group_id.create(m_sub_group_id,"Variables");
  }

  if (m_file_id.isBad()){
    OStringStream ostr;
    ostr() << "Unable to open file <" << m_filename << ">";
    throw ReaderWriterException(func_name,ostr.str());
  }
  if (m_sub_group_id.isBad()){
    OStringStream ostr;
    ostr() << "HDF5 group '" << m_sub_group_name << "' not found";
    throw ReaderWriterException(func_name,ostr.str());
  }
#if 0
  if (m_variable_group_id.isBad()){
    OStringStream ostr;
    ostr() << "Group HDF5 'Variables' not found";
    throw ReaderWriterException(func_name,ostr.str());
  }
#endif

  info() << " INFO END INITIALIZE";


  if (m_open_mode==OpenModeRead){
    int index = 0;
    //H5Giterate(m_sub_group_id.id(),"Variables",&index,_Hdf5MpiReaderWriterIterateMe,this);
    H5Giterate(m_file_id.id(),m_sub_group_name.localstr(),&index,_Hdf5MpiReaderWriterIterateMe,this);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hdf5MpiReaderWriter::
~Hdf5MpiReaderWriter()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_checkValid()
{
  if (m_is_initialized)
    return;
  fatal() << "Use of a Hdf5MpiReaderWriter instance not initialized";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Hdf5MpiReaderWriter::
_variableGroupName(IVariable* var)
{
  return var->fullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
beginWrite(const VariableCollection& vars)
{
  IParallelMng* pm = m_parallel_mng;
  Integer nb_rank = pm->commSize();

  pwarning() << "Implementation of this checkpoint format is not operational yet";
  
  for( VariableCollection::Enumerator i(vars); ++i; ){
    IVariable* v = *i;
    if (v->itemKind()==IK_Unknown)
      continue;

    Ref<ISerializedData> sdata(v->data()->createSerializedDataRef(false));
    Int64 nb_base_element = sdata->nbBaseElement();

    Int64 my_size = nb_base_element;
    Int64ConstArrayView a_my_size(1,&my_size);
    SharedArray<Int64> all_sizes(nb_rank);
    pm->allGather(a_my_size,all_sizes);

    Int64 total_size = 0;
    for( Integer i=0; i<nb_rank; ++i )
      total_size += all_sizes[i];
    Int64 my_index = 0;
    for( Integer i=0; i<m_my_rank; ++i )
      my_index += all_sizes[i];
    m_variables_offset.insert(std::make_pair(v->fullName(),VarOffset(my_index,total_size,all_sizes)));
    info() << " ADD OFFSET v=" << v->fullName() << " offset=" << my_index
           << "  total_size=" << total_size;
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture en parallèle.
 *
 * \warning En cours de test, pas utilisable.
 */
void Hdf5MpiReaderWriter::
_writeValParallel(IVariable* v,const ISerializedData* sdata)
{
  SerializeBuffer sb;
  sb.setMode(ISerializer::ModeReserve);
  sb.reserve(DT_Int32,1); // Pour indiquer la fin des envois
  sb.reserve(v->fullName());
  sb.reserve(m_sub_group_name); //!< Nom du groupe.
  //sb.reserveInteger(1); // Pour le type de données
  //sb.reserveInteger(1); // Pour la dimension
  //v->serialize(&sb,0);
  sdata->serialize(&sb);
  sb.allocateBuffer();
  sb.setMode(ISerializer::ModePut);
  sb.putInt32(1); // Indique qu'il s'agit d'un message non vide
  sb.put(v->fullName());
  sb.put(m_sub_group_name); //!< Nom du groupe.
  //sb.putInteger(v->dataType()); // Pour le type de données
  //sb.putInteger(v->dimension()); // Pour la dimension
  //v->serialize(&sb,0);
  sdata->serialize(&sb);

  m_parallel_mng->sendSerializer(&sb,m_send_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_directReadVal(IVariable* v,IData* data)
{
  _checkValid();

  info() << "DIRECT READ VAL v=" << v->name();
  _readVal(v,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_directWriteVal(IVariable* v,IData* data)
{
  _checkValid();

  Ref<ISerializedData> sdata(data->createSerializedDataRef(false));

  _writeVal(v->fullName(),m_sub_group_name,sdata.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static herr_t
_Hdf5MpiReaderWriterIterateMe(hid_t g,const char* mn,void* ptr)
{
  Hdf5MpiReaderWriter* rw = reinterpret_cast<Hdf5MpiReaderWriter*>(ptr);
  return rw->iterateMe(g,mn);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

herr_t Hdf5MpiReaderWriter::
iterateMe(hid_t group_id,const char* member_name)
{
  ARCANE_UNUSED(group_id);

  m_variables_name.add(std::string_view(member_name));
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
void Hdf5MpiReaderWriter::
_writeVal(const String& var_group_name,const String& sub_group_name,
          const ISerializedData* sdata)
{
  ARCANE_UNUSED(sub_group_name);
  const char* func_name = "Hdf5MpiReaderWriter::_writeVal() ";
  Timer::Sentry ts(&m_io_timer);
  double v0 = ::MPI_Wtime();
  info() << " SDATA name=" << var_group_name << " nb_element=" << sdata->nbElement()
         << " dim=" << sdata->nbDimension() << " datatype=" << sdata->baseDataType()
         << " nb_basic_element=" << sdata->nbBaseElement()
         << " is_multi=" << sdata->isMultiSize()
         << " dimensions_size=" << sdata->extents().size()
         << " memory_size=" << sdata->memorySize()
         << " bytes_size=" << sdata->constBytes().size();

  hid_t save_typeid = m_types.saveType(sdata->baseDataType());
  hid_t trueid = m_types.nativeType(sdata->baseDataType());
  const void* ptr = sdata->constBytes().data();
  Int64 nb_base_element = sdata->nbBaseElement();

  OffsetMap::const_iterator offset_info = m_variables_offset.find(var_group_name);
  if (offset_info==m_variables_offset.end()){
    fatal() << "Can not find offset informations for ->" << var_group_name;
  }
  Int64 nb_element_to_write = nb_base_element;

  //String var_group_name = _variableGroupName(v);
  RealUniqueArray real_array;
  Real3UniqueArray real3_array;
  Real3x3UniqueArray real3x3_array;
  Int32UniqueArray int32_array;
  if (m_is_parallel && m_fileset_size!=1){
    if (m_send_rank==m_my_rank){
      // Je recois les valeurs des autres
      nb_element_to_write = 0;
      for( Integer i=m_send_rank; i<=m_last_recv_rank; ++i ){
        nb_element_to_write += offset_info->second.m_all_sizes[i];
        //info() << "ADD TO WRITE n=" << nb_element_to_write << " add=" << offset_info->second.m_all_sizes[i];
      }
      switch(sdata->baseDataType()){
      case DT_Real:
        real_array.resize(nb_element_to_write);
        ptr = real_array.data();
        break;
      case DT_Real3:
        real3_array.resize(nb_element_to_write);
        ptr = real3_array.data();
        break;
      case DT_Real3x3:
        real3x3_array.resize(nb_element_to_write);
        ptr = real3x3_array.data();
        break;
      case DT_Int32:
        int32_array.resize(nb_element_to_write);
        ptr = int32_array.data();
        break;
      default:
        fatal() << "Type not handled "<< dataTypeName(sdata->baseDataType());
      }
    }
    else{
      return;
      // J'envoie à mon référent
      //switch(sdata->baseDataType()){
      //case DT_Real:
      //_send(sdata,Real());
      // break;
      //}
    }
  }

#if 0
  HGroup var_base_group;
  var_base_group.recursiveCreate(m_file_id,sub_group_name);

  // Création du groupe contenant les informations de la variable
  HGroup group_id;
  //group_id.create(m_variable_group_id,var_group_name);
  group_id.create(var_base_group,var_group_name);
  if (group_id.isBad()){
    OStringStream ostr;
    ostr() << "Group HDF5 '" << var_group_name << "' not found";
    throw ReaderWriterException(func_name,ostr.str());
  }
#endif
  
  //Integer dim2 = dim2_array.size();
  //Integer nb_element = sdata->nbElement();
#if 0
  bool is_multi_size = sdata->isMultiSize();
  Integer dim2_size = 0;
  Integer dim1_size = 0;
  if (nb_dimension==2 && !is_multi_size){
    dim1_size = dimensions[0];
    dim2_size = dimensions[1];
  }
#endif
  //Integer dimension_array_size = dimensions.size();

#if 0
  // Sauve les informations concernant les tailles et dimensions de la variable
  {
    hsize_t att_dims[1];
    att_dims[0] = 9;
    HSpace space_id;
    space_id.createSimple(1,att_dims);
    Integer dim_val[9];

    dim_val[0] = nb_dimension;
    dim_val[1] = dim1_size;
    dim_val[2] = dim2_size;
    dim_val[3] = nb_element;
    dim_val[4] = nb_base_element;
    dim_val[5] = dimension_array_size;
    dim_val[6] = is_multi_size ? 1 : 0;
    dim_val[7] = (Integer)sdata->baseDataType();
    dim_val[8] = sdata->memorySize();

    HAttribute att_id;

    att_id.create(group_id,"Dims",m_types.saveType(dim1_size),space_id);
    herr_t herr = att_id.write(m_types.nativeType(dim2_size),dim_val);
    if (herr<0){
      OStringStream ostr;
      ostr() << "Bad writing of the dimensions for the variable '" << var_group_name << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
  }
#endif

#if 0
  // Si la variable est de type tableau à deux dimensions, sauve les
  // tailles de la deuxième dimension par élément.
  if (dimension_array_size!=0){
    hsize_t att_dims[1];
    att_dims[0] = dimension_array_size;
    HSpace space_id;
    HDataset array_id;

    space_id.createSimple(1,att_dims);

    array_id.create(group_id,"Dim2",m_types.saveType(dim1_size),space_id,H5P_DEFAULT);
    herr_t herr = array_id.write(m_types.nativeType(dim1_size),dimensions.begin());
    if (herr<0){
      OStringStream ostr;
      ostr() << "Bad writing of the dimensions for the variable '" << var_group_name << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
  }
#endif

  //IParallelMng* pm = m_parallel_mng;
  //Integer nb_rank = pm->commSize();

  // Maintenant, sauve les valeurs si necessaire
  if (nb_base_element!=0 && ptr!=0){
    debug(Trace::High) << "Variable " << var_group_name << " begin dumped (nb_base_element=" << nb_base_element << ").";

    hsize_t offset[1];
    hsize_t count[1];
    offset[0] = 0;
    count[0] = nb_element_to_write;

    //Int64UniqueArray all_sizes(nb_rank);
    //Int64 my_size = nb_base_element;
    //Int64ConstArrayView a_my_size(1,&my_size);
    //double v1 = MPI_Wtime();
    //pm->allGather(a_my_size,all_sizes);
    //info() << " CLOCK GATHER = " << (MPI_Wtime() - v1);

    //Int64 total_size = 0;
    //for( Integer i=0; i<nb_rank; ++i )
    //total_size += all_sizes[i];
    //Int64 my_index = 0;
    // for( Integer i=0; i<m_my_rank; ++i )
    //my_index += all_sizes[i];
    //my_index -= nb_base_element;
    Int64 my_index = offset_info->second.m_offset;
    Int64 total_size = offset_info->second.m_total_size;
    offset[0] = my_index;

    double v1 = MPI_Wtime();
    hsize_t dims[1];
    dims[0] = total_size;
    HSpace filespace_id;
    filespace_id.createSimple(1,dims);
    HSpace memspace_id;
    memspace_id.createSimple(1,count);
    if (memspace_id.isBad()){
      OStringStream ostr;
      ostr() << "Wrong dataspace for variable '" << var_group_name << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }


    HDataset dataset_id;

    //hid_t plist_id = H5P_DEFAULT;
    hid_t write_plist_id = H5Pcreate(H5P_DATASET_XFER);
#ifdef H5_HAVE_PARALLEL
    H5Pset_dxpl_mpio(write_plist_id, H5FD_MPIO_COLLECTIVE);
#endif
    //H5Pset_dxpl_mpio(write_plist_id, H5FD_MPIO_INDEPENDENT);
    
    hid_t create_dataset_plist_id = H5P_DEFAULT;
#if 0
    Integer chunk_size = (4096 << 9);
    if (total_size>chunk_size){
      create_dataset_plist_id = H5Pcreate(H5P_DATASET_CREATE);
      H5Pcreate(H5P_DATASET_CREATE);
      hsize_t chunk_dim[1];
      chunk_dim[0] = chunk_size;
      herr_t r = H5Pset_chunk(create_dataset_plist_id,1,chunk_dim);
      info() << " SET CHUNK FOR " << var_group_name << " total=" << total_size << " chunk=" << chunk_dim[0];
    }
#endif

    //dataset_id.create(group_id,"Values",save_typeid,filespace_id,plist_id);
    v1 = MPI_Wtime();
    dataset_id.create(m_variable_group_id,var_group_name,save_typeid,filespace_id,create_dataset_plist_id);
    if (dataset_id.isBad()){
      OStringStream ostr;
      ostr() << "Wrong dataset for variable '" << var_group_name << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
    H5Sselect_hyperslab(filespace_id.id(), H5S_SELECT_SET, offset, NULL, count, NULL);

    
    v1 = MPI_Wtime();
    {
      Timer::Sentry ts(&m_write_timer);
      herr_t herr = dataset_id.write(trueid,ptr,memspace_id,filespace_id,write_plist_id);
      if (herr<0){
        OStringStream ostr;
        ostr() << "Wrong dataset written for variable '" << var_group_name << "'";
        throw ReaderWriterException(func_name,ostr.str());
      }
    }
    if (create_dataset_plist_id!=H5P_DEFAULT)
      H5Pclose(create_dataset_plist_id);
    H5Pclose(write_plist_id);

    info() << " WRITE DATASET name=" << var_group_name
           << " offset=" << offset[0]
           << " mysize=" << nb_base_element
           << " write_size=" << count[0]
           << " total=" << total_size
           << " rank=" << m_my_rank
           << " clock=" << (MPI_Wtime() - v1);

    //pinfo() << " CLOCK WRITE = " <<  << " CPU=" << m_my_rank;
    //pm->barrier();
    //info() << " CLOCK BARRIER = " << (MPI_Wtime() - v1);

    dataset_id.close();
  }
  info() << "TOTAL = " << (MPI_Wtime()-v0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializedData> Hdf5MpiReaderWriter::
_readDim2(IVariable* var)
{
  const char* func_name = "Hdf5MpiReaderWriter::_readDim2()";

  const int max_dim = 256; // Nombre maxi de dimensions des tableaux HDF

  String vname = _variableGroupName(var);

  info() << " READ DIM name=" << vname;

  Integer dimension_array_size = 0;
  Integer nb_element = 0;
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
    if (!is_found){
      OStringStream ostr;
      ostr() << "No HDF5 group with name '" << vname << "' exists";
      throw ReaderWriterException(func_name,ostr.str());
    }
  }

  // Récupère le groupe contenant les informations de la variable
  HGroup group_id;
  //group_id.open(m_variable_group_id,vname);
  group_id.open(m_sub_group_id,vname);
  if (group_id.isBad()){
    OStringStream ostr;
    ostr() << "No HDF5 with name '" << vname << "' exists";
    throw ReaderWriterException(func_name,ostr.str());
  }
  bool is_multi_size = false;
  eDataType data_type = DT_Unknown;
  Integer memory_size = 0;
  Integer nb_base_element = 0;
  Integer dim1_size = 0;
  Integer dim2_size = 0;
  Int64UniqueArray dims;
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

    Integer dim_val[9];
    //herr_t herr = H5Aread(att_id,nativeType(Integer()),dim_val);
    att_id.read(m_types.nativeType(Integer()),dim_val);
    if (hdf_dims[0]!=9){
      OStringStream ostr;
      ostr() << "Wrong dimensions for variable '" << vname
             << "' (found: " << (int)hdf_dims[0] << " expected 9)";
      throw ReaderWriterException(func_name,ostr.str());
    }
    nb_dimension = dim_val[0];
    dim1_size = dim_val[1];
    dim2_size = dim_val[2];
    nb_element = dim_val[3];
    nb_base_element = dim_val[4];
    dimension_array_size = dim_val[5];
    is_multi_size = dim_val[6]!=0;
    data_type = (eDataType)dim_val[7];
    memory_size = dim_val[8];
  }

  info() << " READ DIM name=" << vname
         << " nb_dim=" << nb_dimension << " dim1_size=" << dim1_size
         << " dim2_size=" << dim2_size << " nb_element=" << nb_element
         << " dimension_size=" << dimension_array_size
         << " is_multi_size=" << is_multi_size
         << " data_type" << data_type;

  if (dimension_array_size>0){
    HDataset array_id;
    array_id.open(group_id,"Dim2");
    //hid_t array_id   = H5Dopen(group_id.id(),"Dim2");
    if (array_id.isBad()){
      OStringStream ostr;
      ostr() << "Wrong dataset for variable '" << vname << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
    HSpace space_id = array_id.getSpace();
    if (space_id.isBad()){
      OStringStream ostr;
      ostr() << "Wrong dataspace for variable '" << vname << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
    hsize_t hdf_dims[max_dim];
    hsize_t max_dims[max_dim];
    H5Sget_simple_extent_dims(space_id.id(),hdf_dims,max_dims);
    // Vérifie que le nombre d'éléments du dataset est bien égal à celui
    // attendu.
    if ((Integer)hdf_dims[0]!=dimension_array_size){
      OStringStream ostr;
      ostr() << "Wrong number of elements in 'Dim2' for variable '"
             << vname << "' (found: " << hdf_dims[0]
             << " expected " << dimension_array_size << ")";
      throw ReaderWriterException(func_name,ostr.str());
    }
    dim2_size = 0;
    dims.resize(dimension_array_size);
    herr_t herr = array_id.read(m_types.nativeType(Integer()),dims.data());
    if (herr<0){
      OStringStream ostr;
      ostr() << "Wrong dataset read for variable '" << vname << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }
  }

  Ref<ISerializedData> sdata = arcaneCreateSerializedDataRef(data_type,memory_size,nb_dimension,nb_element,
                                                             nb_base_element,is_multi_size,dims);
  return sdata;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
write(IVariable* v,IData* data)
{
  if (v->itemKind()==IK_Unknown)
    return;
  //if (v->dataType()==DT_Real3)
  //return;
  _directWriteVal(v,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_readVal(IVariable* v,IData* data)
{
  const char* func_name = "Hdf5MpiReaderWriter::_readVal() ";

  String var_group_name = _variableGroupName(v);

  info() << " TRY TO READ var_group=" << var_group_name;

  Ref<ISerializedData> sd(_readDim2(v));
  Int64 storage_size = sd->memorySize();
  //ByteUniqueArray byte_values(storage_size);
  info() << " READ DATA n=" << storage_size;

  data->allocateBufferForSerializedData(sd.get());

  //bool no_dump = v.property() & IVariable::PNoDump;
  // Lit toujours, car le fait de sauver ou non se fait en amont
  //bool no_dump = false;
  if (storage_size!=0){
    // Récupère le groupe contenant les informations de la variable
    HGroup group_id;
    //group_id.open(m_variable_group_id,var_group_name);
    group_id.open(m_sub_group_id,var_group_name);
    if (group_id.isBad()){
      OStringStream ostr;
      ostr() << "No HDF5 group with name '" << var_group_name << "' exists";
      throw ReaderWriterException(func_name,ostr.str());
    }

    HDataset dataset_id;
    dataset_id.open(group_id,"Values");
    if (dataset_id.isBad()){
      OStringStream ostr;
      ostr() << "Wrong dataset for variable '" << var_group_name << "'";
      throw ReaderWriterException(func_name,ostr.str());
    }

    //dataset_id.read(trueid,ptr);
    //debug(Trace::High) << "Variable " << var_group_name << " readed (nb_element=" << nb_element << ").";
    void* ptr = sd->writableBytes().data();
    info() << "READ Variable " << var_group_name << " ptr=" << ptr;;
    hid_t trueid = m_types.nativeType(sd->baseDataType());
    dataset_id.read(trueid,ptr);
  }

  data->assignSerializedData(sd.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
read(IVariable* var,IData* data)
{
  _directReadVal(var,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
setMetaData(const String& meta_data)
{
  ARCANE_UNUSED(meta_data);
#if 0
  if (m_is_parallel){
    IParallelMng* pm = m_parallel_mng;
    Integer nb_rank = pm->commSize();
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
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_setMetaData(const String& meta_data,const String& sub_group_name)
{
  ARCANE_UNUSED(meta_data);
  ARCANE_UNUSED(sub_group_name);
#if 0
  const char* func_name ="Hdf5MpiReaderWriter::setMetaData()";

  HGroup base_group;
  base_group.recursiveCreate(m_file_id,sub_group_name);

  ByteConstArrayView meta_data_utf8 = meta_data.utf8();
  const Byte* _meta_data = meta_data_utf8.begin();

  hsize_t dims[1];
  dims[0] = meta_data_utf8.size() + 1;
  HSpace space_id;
  space_id.createSimple(1,dims);
  if (space_id.isBad())
    throw ReaderWriterException(func_name,"Bad 'space' for the meta-data ('MetaData')");

  HDataset dataset_id;
  dataset_id.create(base_group,"MetaData",m_types.nativeType(Byte()),space_id,H5P_DEFAULT);
  if (dataset_id.isBad())
    throw ReaderWriterException(func_name,"Bad 'dataset' for the meta-data ('MetaData')");

  herr_t herr = dataset_id.write(m_types.nativeType(Byte()),_meta_data);
  if (herr<0)
    throw ReaderWriterException(func_name,"Can't write the meta-data ('MetaData')");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String Hdf5MpiReaderWriter::
metaData()
{
  const char* func_name ="Hdf5MpiReaderWriter::readMetaData()";
  HDataset dataset_id;
  dataset_id.open(m_sub_group_id,"MetaData");
  if (dataset_id.isBad()){
    throw ReaderWriterException(func_name,"Wrong dataset for meta-data ('MetaData')");
  }
  HSpace space_id = dataset_id.getSpace();
  if (space_id.isBad()){
    throw ReaderWriterException(func_name,"Wrong space for meta-data ('MetaData')");
  }
  const int max_dim = 256;
  hsize_t hdf_dims[max_dim];
  hsize_t max_dims[max_dim];
  H5Sget_simple_extent_dims(space_id.id(),hdf_dims,max_dims);
  if (hdf_dims[0]<=0)
    throw ReaderWriterException(func_name,"Wrong number of elements for meta-data ('MetaData')");
  Integer nb_byte = static_cast<Integer>(hdf_dims[0]);
  ByteUniqueArray uchars(nb_byte);
  dataset_id.read(m_types.nativeType(Byte()),uchars.data());
  String s(uchars);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
endWrite()
{
#if 0
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
#endif
  {
    info() << " Hdf5Timer: nb_activated=" << m_io_timer.nbActivated()
           << " time=" << m_io_timer.totalTime()
           << " write=" << m_write_timer.nbActivated()
           << " timewrite=" << m_write_timer.totalTime();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_receiveRemoteVariables()
{
  IParallelMng* pm = m_parallel_mng;
  Integer nb_remaining = m_last_recv_rank - m_send_rank;
  info() << "NB REMAINING = " << nb_remaining;
  Ref<ISerializeMessageList> m_messages(pm->createSerializeMessageListRef());
  
  while(nb_remaining>0){
    ISerializeMessage* sm = new SerializeMessage(m_my_rank,NULL_SUB_DOMAIN_ID,ISerializeMessage::MT_Recv);
    m_messages->addMessage(sm);
    m_messages->processPendingMessages();
    m_messages->waitMessages(Parallel::WaitAll);

    ISerializer* sb = sm->serializer();
    sb->setMode(ISerializer::ModeGet);
    //info() << " RECEIVING BUFFER!";
    Int32 id = sb->getInt32();
    if (id==0){
      //info() << " LAST MESSAGE!";
      --nb_remaining;
    }
    else
      _writeRemoteVariable(sb);
    delete sm;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Hdf5MpiReaderWriter::
_writeRemoteVariable(ISerializer* sb)
{
  String var_name;
  sb->get(var_name);
  String group_name;
  sb->get(group_name);
  //eDataType data_type = (eDataType)sb->getInteger();
  //Integer dim = sb->getInteger();
  //info() << " REMOTE VAR = name=" << var_name << " data_type=" << data_type
  //       << " dim=" << dim << " group=" << group_name;
  Ref<ISerializedData> sdata = arcaneCreateEmptySerializedDataRef();
  sb->setReadMode(ISerializer::ReadReplace);
  sdata->serialize(sb);
  _writeVal(var_name,group_name,sdata.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Protection/reprise au format ArcaneHdf5.
 */
class ArcaneHdf5MpiCheckpointService2
: public ArcaneHdf5MpiReaderWriterObject
{
 public:

  ArcaneHdf5MpiCheckpointService2(const ServiceBuildInfo& sbi)
  : ArcaneHdf5MpiReaderWriterObject(sbi), m_write_index(0), m_writer(0), m_reader(0)
    , m_fileset_size(0)
    {
    }
  virtual IDataWriter* dataWriter() { return m_writer; }
  virtual IDataReader* dataReader() { return m_reader; }

  virtual void notifyBeginWrite();
  virtual void notifyEndWrite();
  virtual void notifyBeginRead();
  virtual void notifyEndRead();
  virtual void close() {}
  virtual String readerServiceName() const { return "ArcaneHdf5MpiCheckpointReader2"; }

 private:

  Integer m_write_index;
  Hdf5MpiReaderWriter* m_writer;
  Hdf5MpiReaderWriter* m_reader;
  Integer m_fileset_size;

 private:

  String _defaultFileName()
    {
      return "arcanedump.mpi.h5";
    }
  Directory _defaultDirectory()
    {
      return Directory(baseDirectoryName());
    }
  void _parseMetaData(String meta_data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5MpiCheckpointService2::
_parseMetaData(String meta_data)
{
  IIOMng* io_mng = subDomain()->ioMng();
  ScopedPtrT<IXmlDocumentHolder> xml_doc(io_mng->parseXmlBuffer(meta_data.utf8(),"MetaData"));
  XmlNode root = xml_doc->documentNode().documentElement();
  Integer version = root.attr("version").valueAsInteger();
  if (version!=1){
    throw ReaderWriterException("ArcaneHdf5MpiCheckpointService2::_parseMetaData","Bad version (expected 1)");
  }
  m_fileset_size = 0;

  info() << " FileSet size=" << m_fileset_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5MpiCheckpointService2::
notifyBeginRead()
{
  String meta_data = readerMetaData();
  _parseMetaData(meta_data);

  info() << " GET META DATA READER " << readerMetaData()
         << " filename=" << fileName();

  if (fileName().null()){
    Directory dump_dir(_defaultDirectory());
    //Directory dump_dir(subDomain()->exportDirectory(),"protection");
    //Directory dump_dir("/tmp/grospelx/");
    setFileName(dump_dir.file(_defaultFileName()));
    //setFileName(dump_dir.file("arcanedump.0.h5"));
    //setFileName(_defaultFileName());
  }
  info() << " READ CHECKPOINT FILENAME = " << fileName();
  StringBuilder sub_group;
  //sub_group = "SubDomain";
  //sub_group += subDomain()->subDomainId();
  //sub_group += "/Index";
  //sub_group += currentIndex();
  sub_group  = "Index";
  sub_group += currentIndex();
  m_reader = new Hdf5MpiReaderWriter(subDomain(),fileName(),sub_group.toString(),0,Hdf5MpiReaderWriter::OpenModeRead);
  m_reader->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5MpiCheckpointService2::
notifyEndRead()
{
  delete m_reader;
  m_reader = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5MpiCheckpointService2::
notifyBeginWrite()
{
  if (options())
    m_fileset_size = options()->filesetSize();

  if (fileName().null()){
    Directory dump_dir(_defaultDirectory());
    //info() << "USE TMP DIRECTORY\n";
    //Directory dump_dir("/tmp/grospelx/");
    //dump_dir.createDirectory();
    setFileName(dump_dir.file(_defaultFileName()));
    //setFileName(_defaultFileName());
  }
  Hdf5MpiReaderWriter::eOpenMode open_mode = Hdf5MpiReaderWriter::OpenModeAppend;
  Integer write_index = checkpointTimes().size();
  --write_index;
  if (write_index==0)
    open_mode = Hdf5MpiReaderWriter::OpenModeTruncate;

  //IParallelMng* pm = subDomain()->parallelMng();
  //Integer sid = pm->commRank();

  StringBuilder sub_group;
  //sub_group = "SubDomain";
  //sub_group += sid;
  //sub_group += "/Index";
  //sub_group += write_index;
  
  sub_group  = "Index";
  sub_group += write_index;
  
  m_writer = new Hdf5MpiReaderWriter(subDomain(),fileName(),sub_group.toString(),m_fileset_size,open_mode);
  m_writer->initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneHdf5MpiCheckpointService2::
notifyEndWrite()
{
  OStringStream ostr;
  ostr() << "<infos version='1'>\n";
  ostr() << " <fileset-size>" << m_fileset_size << "</fileset-size>\n";
  ostr() << "</infos>\n";
  setReaderMetaData(ostr.str());
  ++m_write_index;
  delete m_writer;
  m_writer = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(ArcaneHdf5MpiCheckpointService2,
                                   ICheckpointReader,
                                   ArcaneHdf5MpiCheckpointReader2);

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(ArcaneHdf5MpiCheckpointService2,
                                   ICheckpointWriter,
                                   ArcaneHdf5MpiCheckpointWriter2);

ARCANE_REGISTER_SERVICE_HDF5MPIREADERWRITER(ArcaneHdf5MpiCheckpoint2,
                                            ArcaneHdf5MpiCheckpointService2);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
