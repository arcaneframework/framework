// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5ReaderWriter.h                                          (C) 2000-2020 */
/*                                                                           */
/* Outils de lecture/écriture dans un fichier HDF5.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_HDF5READERWRITER_H
#define ARCANE_STD_HDF5READERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataReader.h"
#include "arcane/IDataWriter.h"

#include "arcane/std/Hdf5Utils.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Lecture/Ecriture au format HDF5.
 
 La version de Hdf5 utilisée est au moins la version 1.4.3.

 En ce qui concerne les réels, on ne supporte que la double précision. Ils sont
 donc stockées sur 8 octets aussi.

 Pour les #Real2, #Real2x2, #Real3 et les #Real3x3, on utilise un type composé.
  
 La structure des informations sauvées est la suivante:
 <ul>
 <li> * toutes les variables sont sauvées dans un groupe qui s'appelle "Variables" * .</li>
 <li> * pour chaque variable, un sous-groupe du nom de la variable est créé. Ce
 sous groupe contient les attributs et datasets suivants:
 <ul>
 <li> * Un \e attribut de nom "Dims" qui est un tableau de 1 ou 2 éléments de type #Integer
 qui contient les informations sur les tailles et dimensions de la variable. Cet attribut
 est \b toujours présent et sert entre autre à déterminer si les deux autres \e datasets
 sont présents. La première valeur (indice 0) est toujours le nombre d'éléments du tableau.
 Si la variable est un tableau à une dimension, il n'y a pas d'autres valeurs. Si le
 tableau est bi-dimensionnel, la deuxième valeur est égale à la taille de la
 première dimension du tableau, les tailles de la deuxième dimensions étant
 données par l'attribut "Dim2".</li>
 <li> * Un \e dataset de nom "Dim2". Ce \e dataset n'est présent que si la variables est du
 genre tableau à deux dimensions, lorsque la première dimension n'est pas nulle et que
 le nombre d'éléments n'est pas nul.
 Dans ce cas, ce \e dataset est un tableau de type #Integer dont la taille est
 égale à celle de la première dimension de la variable et donc chaque valeur est
 égale à la taille de la deuxième dimension.</li>
 <li> * Un \e dataset de nom "Values" contenant les valeurs de la variables. Ce \e dataset
 n'est pas présent dans le cas d'une variable de genre tableau dont le nombre
 d'éléments est nul ou lorsque la variable est temporaire (propriété IVariable::PNoDump). * </li>
 </ul>
 </li>
 </ul>
 
 \todo sauve/relit la liste des groupes d'entités du maillage.

 \warning  * La gestion des lecture/ecriture dans ce format est à l'heure actuelle
 au stade expérimental et ne peut pas être utilisée pour assurer une persistence
 à long terme des données.
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

 private:

  //! Mode parallèle actif: ATTENTION: en cours de test uniquement
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
