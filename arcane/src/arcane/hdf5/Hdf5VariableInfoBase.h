// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Hdf5VariableInfoBase.h                                      (C) 2000-2023 */
/*                                                                           */
/* Linking a variable with an HDF5 file.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_HDF5_HDF5VARIABLEINFOBASE_H
#define ARCANE_HDF5_HDF5VARIABLEINFOBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/hdf5/Hdf5Utils.h"
#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IVariable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Base class for reading or writing a variable.
 */
class ARCANE_HDF5_EXPORT Hdf5VariableInfoBase
{
 public:

  /*!
   * \brief Functor to establish the correspondence between a
   * entity of the current mesh and that of the saved mesh.
   */
  class ICorrespondanceFunctor
  {
   public:

    virtual ~ICorrespondanceFunctor() {}

   public:

    virtual Int64 getOldUniqueId(Int64 uid, Integer index) = 0;
  };

 public:

  static const Integer SAVE_IDS = 1;
  static const Integer SAVE_COORDS = 2;

 protected:

  Hdf5VariableInfoBase()
  : m_correspondance_functor(0)
  {}

 public:

  virtual ~Hdf5VariableInfoBase() {}

 public:

  static Hdf5VariableInfoBase* create(IMesh* mesh, const String& name,
                                      const String& family);
  //! Creates an instance for the variable \a variable.
  static Hdf5VariableInfoBase* create(IVariable* variable);

 public:

  //! Path in the Hdf5 file containing the variable value
  const String& path() const { return m_path; }
  //! Sets the path in the Hdf5 file containing the variable value
  void setPath(const String& path) { m_path = path; }
  virtual void readVariable(Hdf5Utils::HFile& hfile, const String& filename,
                            Hdf5Utils::StandardTypes& st, const String& ids_hpath, IData* data) = 0;
  virtual void writeVariable(Hdf5Utils::HFile& hfile, Hdf5Utils::StandardTypes& st) = 0;
  virtual IVariable* variable() const = 0;

 public:

  void writeGroup(Hdf5Utils::HFile& hfile, Hdf5Utils::StandardTypes& st,
                  const String& hdf_path, Integer save_type);
  void readGroupInfo(Hdf5Utils::HFile& hfile, Hdf5Utils::StandardTypes& st,
                     const String& hdf_path, Int64Array& uids, Real3Array& centers);
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
