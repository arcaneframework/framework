// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableBuildInfo.h                                         (C) 2000-2024 */
/*                                                                           */
/* Informations pour construire une variable.                                */
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
 * \brief Paramètres nécessaires à la construction d'une variable.
 */
class ARCANE_CORE_EXPORT VariableBuildInfo
{
 public:

  // Pour accéder au constructeur par défaut.
  friend class NullVariableBuildInfo;
  // TEMPORAIRE Pour accéder à _subDomain(). A supprimer par la suite.
  friend class VariablePrivate;

 private:

  //! Tag pour un VariableBuildInfo nul.
  struct NullTag
  {};

 public:

  /*!
   * \brief Construit un initialiseur pour une variable.
   *
   * \param name nom de la variable
   * \param m module associé
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IModule* m,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable sans l'associer à
   * un module.
   *
   * \param sub_domain gestionnaire de sous-domaine
   * \param name nom de la variable
   * \param property propriétés de la variable
   */
  VariableBuildInfo(ISubDomain* sub_domain,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable sans l'associer à
   * un module.
   *
   * \param variable_mng gestionnaire de variable
   * \param name nom de la variable
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IVariableMng* variable_mng,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IMesh* mesh,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param property propriétés de la variable
   */
  VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable.
   *
   * \param m module associé
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IModule* m,const String& name,
                    const String& item_family_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IMesh* mesh,const String& name,
                    const String& item_family_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param property propriétés de la variable
   */
  VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,
                    const String& item_family_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable non associée à un maillage.
   *
   * \param sd sous-domaine
   * \param name nom de la variable
   * \param mesh_name nom du maillage
   * \param item_family_name nom de la famille d'entité
   * \param property propriétés de la variable
   */
  VariableBuildInfo(ISubDomain* sd,const String& name, const String& mesh_name,
                    const String& item_family_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable non associée à un maillage.
   *
   * \param variable_mng gestionnaire de variable
   * \param name nom de la variable
   * \param mesh_name nom du maillage
   * \param item_family_name nom de la famille d'entité
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IVariableMng* variable_mng,const String& name, const String& mesh_name,
                    const String& item_family_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param family famille d'entité
   * \param name nom de la variable
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IItemFamily* family,const String& name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable.
   *
   * \param m module associé
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param group_name nom du groupe associé (pour les variables partielles)
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IModule* m,const String& name,
                    const String& item_family_name,
                    const String& item_group_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param group_name nom du groupe associé (pour les variables partielles)
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IMesh* mesh,const String& name,
                    const String& item_family_name,
                    const String& item_group_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
   *
   * \param mesh maillage
   * \param name nom de la variable
   * \param item_family_name nom de la famille d'entité
   * \param group_name nom du groupe associé (pour les variables partielles)
   * \param property propriétés de la variable
   */
  VariableBuildInfo(const MeshHandle& mesh_handle,const String& name,
                    const String& item_family_name,
                    const String& item_group_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable non associée à un maillage
   *
   * \param sd sous-domaine
   * \param name nom de la variable
   * \param mesh_name nom du maillage
   * \param item_family_name nom de la famille d'entité
   * \param group_name nom du groupe associé (pour les variables partielles)
   * \param property propriétés de la variable
   */
  VariableBuildInfo(ISubDomain* sd,const String& name,
                    const String& mesh_name,
                    const String& item_family_name,
                    const String& item_group_name,int property=0);

  /*!
   * \brief Construit un initialiseur pour une variable non associée à un maillage
   *
   * \param variable_mng gestionnaire de variable
   * \param name nom de la variable
   * \param mesh_name nom du maillage
   * \param item_family_name nom de la famille d'entité
   * \param group_name nom du groupe associé (pour les variables partielles)
   * \param property propriétés de la variable
   */
  VariableBuildInfo(IVariableMng* variable_mng,const String& name,
                    const String& mesh_name,
                    const String& item_family_name,
                    const String& item_group_name,int property=0);

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

  ISubDomain* m_sub_domain = nullptr; //!< Gestionnaire de sous-domaine
  IModule* m_module = nullptr; //!< Module associé à la variable
  MeshHandle m_mesh_handle;  //!< Handle sur le maillage
  String m_name; //!< Nom de la variable
  String m_item_family_name; //!< Nom de la famille d'entité
  String m_item_group_name; //!< Nom du groupe d'entité support
  String m_mesh_name; //!< Nom du maillage associé à la variable
  int m_property = 0; //!< Propriétés de la variable
  bool m_is_null = false;

 private:

  void _init();
  ISubDomain* _subDomain() const { return m_sub_domain; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Constructeur pour une variable nulle.
 *
 * \warning Cette classe est expérimentale. Ne pas utiliser en dehors
 * de Arcane.
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

