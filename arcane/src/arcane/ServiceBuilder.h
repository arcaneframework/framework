// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuilder.h                                            (C) 2000-2019 */
/*                                                                           */
/* Classe utilitaire pour instantier un service.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEBUILDER_H
#define ARCANE_SERVICEBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"

#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/IApplication.h"
#include "arcane/IMesh.h"
#include "arcane/ICaseOptions.h"
#include "arcane/IFactoryService.h"
#include "arcane/ServiceFinder2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Propriétés pour la création de service.
 *
 * Il s'agit de drapeaux qui s'utilisent avec l'opérateur ou binaire (|)
 */
enum eServiceBuilderProperties
{
  //! Aucune propriété particulière
  SB_None = 0,
  //! Autorise l'absence du service
  SB_AllowNull = 1,
  //! Indique que tous les processus font la même opération
  SB_Collective = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Service
 * \brief Classe utilitaire pour instantier un service d'une interface donnée.
 *
 * Cette classe permet de rechercher l'ensemble des services disponibles
 * et implémentant l'interface \a InterfaceType passée en paramètre template.
 *
 * Cette classe remplace les anciennes classes qui permettaient de créer
 * des services, à savoir ServiceFinderT, ServiceFinder2T et FactoryT.
 *
 * Il existe trois constructeurs suivant que l'on souhaite instantier
 * un service du sous-domaine, de la session ou de l'application. En général,
 * il s'agit d'un service de sous-domaine, les deux dernières catégories
 * étant plutôt utilisées pour les services internes à Arcane.
 *
 * L'exemple suivant créé un service de sous-domaine implémentant
 * l'interface \a IMyInterface et de nom \a TOTO:
 * \code
 * ISubDomain* sd  = ...
 * ServiceBuilder<IMyInterface> builder(sd);
 * ServiceRef<IMyInterface> iservice = builder.createReference("TOTO");
 * ...
 * \endcode
 *
 * L'instance retournée est gérée par compteur de référence et est détruite
 * dès qu'il n'y a plus de référence dessus.
 * Par défaut, createInstance() lève une exception si le service n'est pas
 * trouvé, sauf si la propriété \a SB_AllowNull est spécifiée..
 * Si la propriété \a SB_Collective est vrai, l'exception levée est du type
 * ParallelFatalErrorException, sinon elle du type FatalErrorException.
 * Cela est utile si on est sur
 * que tous les processus vont faire la même opération. Dans ce cas,
 * cela permet de ne générer qu'un seul message d'erreur et d'arrêter
 * le code proprement.
 *
 * Il est aussi possible de récupérer une instance singleton d'un service,
 * via getSingleton(). Les instances singletons qui sont disponibles
 * sont référencées dans le fichier de configuration du code (voir \ref arcanedoc_codeconfig).
 */
template<typename InterfaceType>
class ServiceBuilder
{
 public:

  //! Instantiation pour créer un service d'un sous-domaine.
  ServiceBuilder(ISubDomain* sd)
  : m_service_finder(sd->application(),ServiceBuildInfoBase(sd))
  {}
  //! Instantiation pour créer un service d'une session.
  ServiceBuilder(ISession* session)
  : m_service_finder(session->application(),ServiceBuildInfoBase(session))
  {}
  //! Instantiation pour créer un service de l'application.
  ServiceBuilder(IApplication* app)
  : m_service_finder(app,ServiceBuildInfoBase(app))
  {}
  //! Instantiation pour créer un service d'option du jeu de données
  ServiceBuilder(IApplication* app,ICaseOptions* opt)
  : m_service_finder(app,ServiceBuildInfoBase(_arcaneDeprecatedGetSubDomain(opt),opt))
  {}
  
  ~ServiceBuilder(){ }

 public:
  
  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   *
   * L'instance est créée avec la fabrique enregistrée sous le nom \a name.
   *
   * Par défaut, une exception est levée si le service spécifiée n'est pas trouvé.
   * Il est possible de changer ce comportement en spécifiant SB_AllowNull dans \a properties
   * auquel cas la fonction retourne un pointeur nul si le service spécifié n'existe pas.
   */
  Ref<InterfaceType>
  createReference(const String& name,eServiceBuilderProperties properties=SB_None)
  {
    Ref<InterfaceType> mf = m_service_finder.createReference(name);
    if (!mf){
      if (properties & SB_AllowNull)
        return {};
      _throwFatal(name,properties);
    }
    return mf;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   *
   * L'instance est créée avec la fabrique enregistrée sous le nom \a name.
   * Le pointeur retourné doit être désalloué par delete.
   *
   * Il est possible de spécifier le maillage \a mesh sur lequel reposera le service.
   * Cela n'est utile que pour les services de sous-domaine. Pour les services
   * de session ou d'application, cet argument n'est pas utilisé.
   *
   * Par défaut, une exception est levée si le service spécifiée n'est pas trouvé.
   * Il est possible de changer ce comportement en spécifiant SB_AllowNull dans \a properties
   * auquel cas la fonction retourne un pointeur nul si le service spécifié n'existe pas.
   */
  Ref<InterfaceType>
  createReference(const String& name,IMesh* mesh,
                  eServiceBuilderProperties properties=SB_None)
  {
    Ref<InterfaceType> mf = m_service_finder.createReference(name,mesh);
    if (!mf){
      if (properties & SB_AllowNull)
        return {};
      _throwFatal(name,properties);
    }
    return mf;
  }

  /*!
   * \brief Créé une instance de chaque service qui implémente \a InterfaceType.
   *
   * Les instances créées sont rangées dans \a instances. L'appelant doit les
   * détruire via l'opérateur delete une fois qu'elles ne sont plus utiles.
   */
  UniqueArray<Ref<InterfaceType>> createAllInstances()
  {
    return m_service_finder.createAll();
  }

  /*!
   * \brief Instance singleton du service implémentant l'interface \a InterfaceType.
   *
   * L'instance retournée ne doit pas être détruite.
   *
   * Par défaut, une exception est levée si le service spécifiée n'est pas trouvé.
   * Il est possible de changer ce comportement en spécifiant SB_AllowNull dans \a properties
   * auquel cas la fonction retourne un pointeur nul si le service spécifié n'existe pas.
   */
  InterfaceType* getSingleton(eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* mf = m_service_finder.getSingleton();
    if (!mf){
      if (properties & SB_AllowNull)
        return 0;
      _throwFatal(properties);
    }
    return mf;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  static Ref<InterfaceType>
  createReference(ISubDomain* sd,const String& name,
                  eServiceBuilderProperties properties=SB_None)
  {
    return createReference(sd,name,0,properties);
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  static Ref<InterfaceType>
  createReference(ISession* session,const String& name,
                  eServiceBuilderProperties properties=SB_None)
  {
    Ref<InterfaceType> it;
    {
      ServiceBuilder sb(session);
      it = sb.createReference(name,properties);
    }
    return it;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  static Ref<InterfaceType>
  createReference(IApplication* app,const String& name,
                  eServiceBuilderProperties properties=SB_None)
  {
    Ref<InterfaceType> it;
    {
      ServiceBuilder sb(app);
      it = sb.createReference(name,properties);
    }
    return it;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  static Ref<InterfaceType>
  createReference(ISubDomain* sd,const String& name,IMesh* mesh,
                  eServiceBuilderProperties properties=SB_None)
  {
    Ref<InterfaceType> it;
    {
      ServiceBuilder sb(sd);
      it = sb.createReference(name,mesh,properties);
    }
    return it;
  }

  //! Remplit \a names avec les noms des services disponibles pour instantier cette interface
  void getServicesNames(Array<String>& names) const
  {
    m_service_finder.getServicesNames(names);
  }

 public:
  /*!
   * \brief Créé une instance de chaque service qui implémente \a InterfaceType.
   *
   * Les instances créées sont rangées dans \a instances. L'appelant doit les
   * détruire via l'opérateur delete une fois qu'elles ne sont plus utiles.
   *
   * \deprecated Utilise la surcharge qui retourne un tableau de références.
   */
  ARCCORE_DEPRECATED_2019("use createAllInstances(Array<Ref<InterfaceType>>) instead")
  void createAllInstances(Array<InterfaceType*>& instances)
  {
    m_service_finder.createAll(instances);
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   *
   * L'instance est créée avec la fabrique enregistrée sous le nom \a name.
   * Le pointeur retourné doit être désalloué par delete.
   *
   * Par défaut, une exception est levée si le service spécifiée n'est pas trouvé.
   * Il est possible de changer ce comportement en spécifiant SB_AllowNull dans \a properties
   * auquel cas la fonction retourne un pointeur nul si le service spécifié n'existe pas.
   *
   * \deprecated Utilise createReference() à la place.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  InterfaceType* createInstance(const String& name,eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* mf = m_service_finder.create(name);
    if (!mf){
      if (properties & SB_AllowNull)
        return 0;
      _throwFatal(name,properties);
    }
    return mf;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   * \deprecated Utilise createReference() à la place.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  static InterfaceType*
  createInstance(ISubDomain* sd,const String& name,
                 eServiceBuilderProperties properties=SB_None)
  {
    return createInstance(sd,name,0,properties);
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   *
   * L'instance est créée avec la fabrique enregistrée sous le nom \a name.
   * Le pointeur retourné doit être désalloué par delete.
   *
   * Il est possible de spécifier le maillage \a mesh sur lequel reposera le service.
   * Cela n'est utile que pour les services de sous-domaine. Pour les services
   * de session ou d'application, cet argument n'est pas utilisé.
   *
   * Par défaut, une exception est levée si le service spécifiée n'est pas trouvé.
   * Il est possible de changer ce comportement en spécifiant SB_AllowNull dans \a properties
   * auquel cas la fonction retourne un pointeur nul si le service spécifié n'existe pas.
   *
   * \deprecated Utilise createReference() à la place.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  InterfaceType* createInstance(const String& name,IMesh* mesh,
                                eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* mf = m_service_finder.create(name,mesh);
    if (!mf){
      if (properties & SB_AllowNull)
        return 0;
      _throwFatal(name,properties);
    }
    return mf;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  static InterfaceType* createInstance(ISession* session,const String& name,
                                       eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* it = 0;
    {
      ServiceBuilder sb(session);
      it = sb.createInstance(name,properties);
    }
    return it;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  static InterfaceType* createInstance(IApplication* app,const String& name,
                                       eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* it = 0;
    {
      ServiceBuilder sb(app);
      it = sb.createInstance(name,properties);
    }
    return it;
  }

  /*!
   * \brief Créé une instance implémentant l'interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  static InterfaceType* createInstance(ISubDomain* sd,const String& name,IMesh* mesh,
                                       eServiceBuilderProperties properties=SB_None)
  {
    InterfaceType* it = 0;
    {
      ServiceBuilder sb(sd);
      it = sb.createInstance(name,mesh,properties);
    }
    return it;
  }

 private:

  Internal::ServiceFinderBase2T<InterfaceType> m_service_finder;

 private:
  
  String _getErrorMessage(String wanted_name)
  {
    StringUniqueArray valid_names;
    m_service_finder.getServicesNames(valid_names);
    if (valid_names.size()!=0)
      return String::format("no service named '{0}' found (valid values = {1})",
                            wanted_name,String::join(", ",valid_names));
    // Aucun service disponible
    return String::format("no service named '{0}' found and no implementation available",
                          wanted_name);
  }

  void _throwFatal(const String& name,eServiceBuilderProperties properties)
  {
      String err_msg = _getErrorMessage(name);
      if (properties & SB_Collective)
        throw ParallelFatalErrorException(A_FUNCINFO,err_msg);
      else
        throw FatalErrorException(A_FUNCINFO,err_msg);
  }
  void _throwFatal(eServiceBuilderProperties properties)
  {
    String err_msg = "No singleton service found for that interface";
    if (properties & SB_Collective)
      throw ParallelFatalErrorException(A_FUNCINFO,err_msg);
    else
      throw FatalErrorException(A_FUNCINFO,err_msg);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

