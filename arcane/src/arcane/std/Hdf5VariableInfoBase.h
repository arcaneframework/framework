// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5VariableInfoBase.h                                      (C) 2000-2010 */
/*                                                                           */
/* Liaison d'une variable avec un fichier HDF5.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_HDF5VARIABLEINFOBASE_H
#define ARCANE_STD_HDF5VARIABLEINFOBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/std/Hdf5Utils.h"
#include "arcane/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IVariable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base pour lire ou écrire une variables.
 */
class ARCANE_STD_EXPORT Hdf5VariableInfoBase
{
 public:
  /*!
   * \brief Fonctor pour faire la correspondance entre une
   * entité du maillage courant et celle du maillage sauvegardé.
   */
  class ICorrespondanceFunctor
  {
   public:
    virtual ~ICorrespondanceFunctor() {}
   public:
    virtual Int64 getOldUniqueId(Int64 uid,Integer index) =0;
  };
 public:
  static const Integer SAVE_IDS = 1;
  static const Integer SAVE_COORDS = 2;
 protected:
  Hdf5VariableInfoBase() : m_correspondance_functor(0) {}
 public:
  virtual ~Hdf5VariableInfoBase() {}
 public:
  static Hdf5VariableInfoBase* create(IMesh* mesh,const String& name,
                                      const String& family);
  //! Créé une instance pour la variable \a variable.
  static Hdf5VariableInfoBase* create(IVariable* variable);
                                          
 public:
  //! Chemin dans le fichier Hdf5 contenant la valeur de la variable
  const String& path() const { return m_path; }
  //! Positionne le chemin dans le fichier Hdf5 contenant la valeur de la variable
  void setPath(const String& path) { m_path = path; }
  virtual void readVariable(Hdf5Utils::HFile& hfile,const String& filename,
                            Hdf5Utils::StandardTypes& st,const String& ids_hpath,IData* data) =0;
  virtual void writeVariable(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st) =0;
  virtual IVariable* variable() const =0;
 public:
  void writeGroup(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st,
                  const String& hdf_path,Integer save_type);
  void readGroupInfo(Hdf5Utils::HFile& hfile,Hdf5Utils::StandardTypes& st,
                     const String& hdf_path,Int64Array& uids,Real3Array& centers);
  void setCorrespondanceFunctor(ICorrespondanceFunctor* functor)
  {
    m_correspondance_functor = functor;
  }
 private:
  String m_path;
 protected:
  ICorrespondanceFunctor* m_correspondance_functor;
 private:
  static void _checkValidVariable(IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
