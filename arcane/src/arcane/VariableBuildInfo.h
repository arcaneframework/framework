// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableBuildInfo.h                                         (C) 2000-2021 */
/*                                                                           */
/* Informations pour construire une variable.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLEBUILDINFO_H
#define ARCANE_VARIABLEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/MeshHandle.h"

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
  // TEMPORAIRE Pour accéder à _subDomain(). A supprimer par la suite.
  friend class VariablePrivate;
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
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
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
   * \brief Construit un initialiseur pour une variable associée  à
   * un maillage autre que le maillage par défaut.
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

 private:

  ISubDomain* m_sub_domain; //!< Gestionnaire de sous-domaine
  IModule* m_module; //!< Module associé à la variable
  MeshHandle m_mesh_handle;  //!< Handle sur le maillage
  String m_name; //!< Nom de la variable
  String m_item_family_name; //!< Nom de la famille d'entité
  String m_item_group_name; //!< Nom du groupe d'entité support
  String m_mesh_name; //!< Nom du maillage associé à la variable
  int m_property; //!< Propriétés de la variable

 private:

  void _init();
  ISubDomain* _subDomain() const { return m_sub_domain; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

