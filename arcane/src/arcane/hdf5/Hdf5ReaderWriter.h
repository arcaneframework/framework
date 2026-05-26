// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5ReaderWriter.h                                          (C) 2000-2023 */
/*                                                                           */
/* Tools for reading/writing in an HDF5 file.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HDF5_HDF5READERWRITER_H
#define ARCANE_HDF5_HDF5READERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataReader.h"
#include "arcane/IDataWriter.h"

#include "arcane/hdf5/Hdf5Utils.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Reading/Writing in HDF5 format.
 
 The HDF5 version used is at least version 1.4.3.

 Regarding real numbers, only double precision is supported. They are
 therefore stored in 8 bytes as well.

 For #Real2, #Real2x2, #Real3, and #Real3x3, a composite type is used.
  
 The structure of the saved information is as follows:
 <ul>
 <li> * all variables are saved in a group called "Variables". * .</li>
 <li> * for each variable, a subgroup named after the variable is created. This
 subgroup contains the following attributes and datasets:
 <ul>
 <li> * An attribute named "Dims" which is an array of 1 or 2 #Integer
 elements containing information about the sizes and dimensions of the
 variable. This attribute is always present and is used, among other
 things, to determine if the other two datasets are present. The first value
 (index 0) is always the number of elements in the array. If the variable is a
 one-dimensional array, there are no other values. If the array is
 two-dimensional, the second value is equal to the size of the first dimension
 of the array, while the sizes of the second dimensions are given by the
 "Dim2" attribute.</li>
 <li> * A dataset named "Dim2". This dataset is only present if the variable
 is a two-dimensional array, when the first dimension is not zero and the
 number of elements is not zero. In this case, this dataset is an array of
 #Integer type whose size is equal to the size of the first dimension of the
 variable, and thus each value is equal to the size of the second dimension.
 </li>
 <li> * A dataset named "Values" containing the values of the variable. This
 dataset is not present in the case of an array variable whose number of
 elements is zero or when the variable is temporary (IVariable::PNoDump
 property). * </li>
 </ul>
 </li>
 </ul>
 
 \todo save/read the list of mesh entity groups.

 \warning  * The handling of reading/writing in this format is currently at
 the experimental stage and cannot be used to ensure long-term data persistence.
 */
class Hdf5ReaderWriter
: public TraceAccessor
, public IDataReader
, public IDataWriter
{

 public:

  enum eOpenMode
  {
    OpenModeRead,
    OpenModeTruncate,
    OpenModeAppend
  };
 public:

  Hdf5ReaderWriter(ISubDomain* sd,const String& filename,const String& m_sub_group_name,
                   Integer fileset_size,
                   Integer write_index, Integer index_modulo,
                   eOpenMode om,bool do_verif=false);
  ~Hdf5ReaderWriter();

 public:

  virtual void initialize();

  virtual void beginWrite(const VariableCollection& vars)
  {
    ARCANE_UNUSED(vars);
  }
  virtual void endWrite();
  virtual void beginRead(const VariableCollection& vars)
  {
    ARCANE_UNUSED(vars);
  }
  virtual void endRead() {}

  virtual void setMetaData(const String& meta_data);
  virtual String metaData();

  virtual void write(IVariable* v,IData* data);
  virtual void read(IVariable* v,IData* data);

 public:
	
  herr_t iterateMe(hid_t group_id,const char* member_name);

 private:
	
  IParallelMng* m_parallel_mng; //!< Parallelism manager;
  eOpenMode m_open_mode; //!< Open mode
  String m_filename; //!< Filename.
  String m_sub_group_name; //!< Subgroup name.
  bool m_is_initialized; //!< True if already initialized

  Hdf5Utils::StandardTypes m_types;

  Hdf5Utils::HFile m_file_id;       //!< HDF file identifier 
  Hdf5Utils::HGroup m_sub_group_id; //!< HDF group identifier containing the protection
  Hdf5Utils::HGroup m_variable_group_id; //!< HDF group identifier containing the variables

  StringList m_variables_name; //!< List of names of saved variables.
  Timer m_io_timer;

 private:

  //! Active parallel mode: WARNING: for testing only
  bool m_is_parallel;
  Int32 m_my_rank;
  Int32 m_send_rank;
  Int32 m_last_recv_rank;

  Integer m_fileset_size;
  Integer m_index_write;
  Integer m_index_modulo;

 private:

  void _writeVal(const String& var_group_name,
                 const String& sub_group_name,
                 const ISerializedData* sdata,
                 const Int32 from_rank=0);
  void _writeValParallel(IVariable* v,const ISerializedData* sdata);
  void _readVal(IVariable* var,IData* data);

  Ref<ISerializedData> _readDim2(IVariable* v);

  void _directReadVal(IVariable* v,IData* data);
  void _directWriteVal(IVariable* v,IData* data);
  void _checkValid();
  String _variableGroupName(IVariable* var);

  void _receiveRemoteVariables();
  void _writeRemoteVariable(ISerializer* sb);
  void _setMetaData(const String& meta_data,const String& sub_group_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
