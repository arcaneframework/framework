// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableBuildInfo.h                                         (C) 2000-2024 */
/*                                                                           */
/* Information for building a variable.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEBUILDINFO_H
#define ARCANE_VARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IModule;
class ISubDomain;
class IMesh;
class IItemFamily;
class IVariableMng;
class VariablePrivate;
class IDataFactoryMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parameters necessary for building a variable.
 */
class ARCANE_CORE_EXPORT VariableBuildInfo
{
 public:

  // To access the default constructor.
  friend class NullVariableBuildInfo;
  // TEMPORARY To access _subDomain(). To be removed later.
  friend class VariablePrivate;

 private:

  //! Tag for a null VariableBuildInfo.
  struct NullTag
  {};

 public:

  /*!
   * \brief Constructs an initializer for a variable.
   *
   * \param name variable name
   * \param m associated module
   * \param property variable properties
   */
  VariableBuildInfo(IModule* m, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable without associating it with
   * a module.
   *
   * \param sub_domain subdomain manager
   * \param name variable name
   * \param property variable properties
   */
  VariableBuildInfo(ISubDomain* sub_domain, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable without associating it with
   * a module.
   *
   * \param variable_mng variable manager
   * \param name variable name
   * \param property variable properties
   */
  VariableBuildInfo(IVariableMng* variable_mng, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh other
   * than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param property variable properties
   */
  VariableBuildInfo(IMesh* mesh, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh other
   * than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param property variable properties
   */
  VariableBuildInfo(const MeshHandle& mesh_handle, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable.
   *
   * \param m associated module
   * \param name variable name
   * \param item_family_name entity family name
   * \param property variable properties
   */
  VariableBuildInfo(IModule* m, const String& name,
                    const String& item_family_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh
   * other than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param item_family_name entity family name
   * \param property variable properties
   */
  VariableBuildInfo(IMesh* mesh, const String& name,
                    const String& item_family_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh
   * other than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param item_family_name entity family name
   * \param property variable properties
   */
  VariableBuildInfo(const MeshHandle& mesh_handle, const String& name,
                    const String& item_family_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable not associated with a mesh.
   *
   * \param sd subdomain
   * \param name variable name
   * \param mesh_name mesh name
   * \param item_family_name entity family name
   * \param property variable properties
   */
  VariableBuildInfo(ISubDomain* sd, const String& name, const String& mesh_name,
                    const String& item_family_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable not associated with a mesh.
   *
   * \param variable_mng variable manager
   * \param name variable name
   * \param mesh_name mesh name
   * \param item_family_name entity family name
   * \param property variable properties
   */
  VariableBuildInfo(IVariableMng* variable_mng, const String& name, const String& mesh_name,
                    const String& item_family_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh
   * other than the default mesh.
   *
   * \param family entity family
   * \param name variable name
   * \param property variable properties
   */
  VariableBuildInfo(IItemFamily* family, const String& name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable.
   *
   * \param m associated module
   * \param name variable name
   * \param item_family_name entity family name
   * \param item_group_name associated group name (for partial variables)
   * \param property variable properties
   */
  VariableBuildInfo(IModule* m, const String& name,
                    const String& item_family_name,
                    const String& item_group_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh
   * other than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param item_family_name entity family name
   * \param group_name associated group name (for partial variables)
   * \param property variable properties
   */
  VariableBuildInfo(IMesh* mesh, const String& name,
                    const String& item_family_name,
                    const String& item_group_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable associated with a mesh
   * other than the default mesh.
   *
   * \param mesh mesh
   * \param name variable name
   * \param item_family_name entity family name
   * \param group_name associated group name (for partial variables)
   * \param property variable properties
   */
  VariableBuildInfo(const MeshHandle& mesh_handle, const String& name,
                    const String& item_family_name,
                    const String& item_group_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable not associated with a mesh
   *
   * \param sd subdomain
   * \param name variable name
   * \param mesh_name mesh name
   * \param item_family_name entity family name
   * \param group_name associated group name (for partial variables)
   * \param property variable properties
   */
  VariableBuildInfo(ISubDomain* sd, const String& name,
                    const String& mesh_name,
                    const String& item_family_name,
                    const String& item_group_name, int property = 0);

  /*!
   * \brief Constructs an initializer for a variable not associated with a mesh
   *
   * \param variable_mng variable manager
   * \param name variable name
   * \param mesh_name mesh name
   * \param item_family_name entity family name
   * \param group_name associated group name (for partial variables)
   * \param property variable properties
   */
  VariableBuildInfo(IVariableMng* variable_mng, const String& name,
                    const String& mesh_name,
                    const String& item_family_name,
                    const String& item_group_name, int property = 0);

 private:

  explicit VariableBuildInfo(const NullTag&);

 public:

  ARCCORE_DEPRECATED_2020("Do not use this method. Try to get ISubDomain from another way")
  ISubDomain* subDomain() const { return m_sub_domain; }

 public:

  IVariableMng* variableMng() const;
  IDataFactoryMng* dataFactoryMng() const;
  ITraceMng* traceMng() const;
  IModule* module() const { return m_module; }
  IMesh* mesh() const { return m_mesh_handle.mesh(); }
  const MeshHandle& meshHandle() const { return m_mesh_handle; }
  const String& name() const { return m_name; }
  const String& itemFamilyName() const { return m_item_family_name; }
  const String& itemGroupName() const { return m_item_group_name; }
  const String& meshName() const { return m_mesh_name; }
  int property() const { return m_property; }
  bool isNull() const { return m_is_null; }

 private:

  ISubDomain* m_sub_domain = nullptr; //!< Subdomain manager
  IModule* m_module = nullptr; //!< Module associated with the variable
  MeshHandle m_mesh_handle; //!< Handle on the mesh
  String m_name; //!< Variable name
  String m_item_family_name; //!< Entity family name
  String m_item_group_name; //!< Supported entity group name
  String m_mesh_name; //!< Name of the mesh associated with the variable
  int m_property = 0; //!< Variable properties
  bool m_is_null = false;

 private:

  void _init();
  ISubDomain* _subDomain() const { return m_sub_domain; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Constructor for a null variable.
 *
 * \warning This class is experimental. Do not use it outside of Arcane.
 */
class ARCANE_CORE_EXPORT NullVariableBuildInfo
: public VariableBuildInfo
{
 public:

  NullVariableBuildInfo()
  : VariableBuildInfo(NullTag{})
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
