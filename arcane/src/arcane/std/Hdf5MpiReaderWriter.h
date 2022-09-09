// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5MpiReaderWriter.h                                       (C) 2000-2020 */
/*                                                                           */
/* Outils de lecture/écriture dans un fichier HDF5.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_HDF5MPIREADERWRITER_H
#define ARCANE_STD_HDF5MPIREADERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataReader.h"
#include "arcane/IDataWriter.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/VariableTypes.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Lecture/Ecriture au format HDF5 parallele.
 
 \warning La gestion des lecture/ecriture dans ce format est à l'heure actuelle
 au stade expérimental et ne peut pas être utilisée pour assurer une persistence
 à long terme des données.
 */
class Hdf5MpiReaderWriter
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

  Hdf5MpiReaderWriter(ISubDomain* sd,const String& filename,const String& m_sub_group_name,
                      Integer fileset_size,eOpenMode om,bool do_verif=false);
  ~Hdf5MpiReaderWriter();

 public:

  virtual void initialize();

  virtual void beginWrite(const VariableCollection& vars);
  virtual void endWrite();
  virtual void beginRead(const VariableCollection&) {}
  virtual void endRead() {}

  virtual void setMetaData(const String& meta_data);
  virtual String metaData();

  virtual void write(IVariable* v,IData* data);
  virtual void read(IVariable* v,IData* data);

 public:
	
  herr_t iterateMe(hid_t group_id,const char* member_name);

 private:

  class VarOffset
  {
  public:
    VarOffset(Int64 offset,Int64 total_size,SharedArray<Int64> all_sizes)
      : m_offset(offset), m_total_size(total_size), m_all_sizes(all_sizes)
    {
    }
  public:
    Int64 m_offset;
    Int64 m_total_size;
    SharedArray<Int64> m_all_sizes;
  };
  
  ISubDomain* m_sub_domain; //!< Gestionnaire du sous-domaine
  IParallelMng* m_parallel_mng; //!< Gestionnaire du parallélisme;
  eOpenMode m_open_mode; //!< Mode d'ouverture
  String m_filename; //!< Nom du fichier.
  String m_sub_group_name; //!< Nom du fichier.
  bool m_is_initialized; //!< Vrai si déjà initialisé

  Hdf5Utils::StandardTypes m_types;

  Hdf5Utils::HFile m_file_id;       //!< Identifiant HDF du fichier 
  Hdf5Utils::HGroup m_sub_group_id; //!< Identifiant HDF du groupe contenant la protection
  Hdf5Utils::HGroup m_variable_group_id; //!< Identifiant HDF du groupe contenant les variables

  StringList m_variables_name; //!< Liste des noms des variables sauvées.
  Timer m_io_timer;
  Timer m_write_timer;

  typedef std::map<String,VarOffset> OffsetMap;
  OffsetMap m_variables_offset;

 private:

  //! Mode parallèle actif: ATTENTION: en cours de test uniquement
  bool m_is_parallel;
  Int32 m_my_rank;
  Int32 m_send_rank;
  Int32 m_last_recv_rank;

  Integer m_fileset_size;

 private:

  void _writeVal(const String& var_group_name,const String& sub_group_name,
                 const ISerializedData* sdata);
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
