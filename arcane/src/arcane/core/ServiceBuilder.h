// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceBuilder.h                                            (C) 2000-2023 */
/*                                                                           */
/* Utility class for instantiating a service.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SERVICEBUILDER_H
#define ARCANE_CORE_SERVICEBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"

#include "arcane/core/ISession.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/IFactoryService.h"
#include "arcane/core/ServiceFinder2.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace AxlOptionsBuilder
{
class Document;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Properties for service creation.
 *
 * These are flags used with the binary OR operator (|).
 */
enum eServiceBuilderProperties
{
  //! No specific property.
  SB_None = 0,
  //! Allows the service to be absent.
  SB_AllowNull = 1,
  //! Indicates that all processes perform the same operation.
  SB_Collective = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Service
 * \brief Utility class for instantiating a service of a given interface.
 *
 * This class allows searching for all available services
 * implementing the \a InterfaceType interface passed as a template parameter.
 *
 * This class replaces older classes that allowed service creation,
 * namely ServiceFinderT, ServiceFinder2T, and FactoryT.
 *
 * There are three constructors depending on whether you want to instantiate
 * a subdomain, a session, or an application service. Generally,
 * it is a subdomain service; the last two categories
 * are rather used for internal Arcane services.
 *
 * The following example creates a subdomain service implementing
 * the \a IMyInterface interface with the name \a TOTO:
 * \code
 * ISubDomain* sd  = ...
 * ServiceBuilder<IMyInterface> builder(sd);
 * ServiceRef<IMyInterface> iservice = builder.createReference("TOTO");
 * ...
 * \endcode
 *
 * The returned instance is managed by a reference counter and is destroyed
 * as soon as there are no more references to it.
 * By default, createInstance() throws an exception if the service is not
 * found, unless the \a SB_AllowNull property is specified.
 * If the \a SB_Collective property is true, the exception thrown is of type
 * ParallelFatalErrorException; otherwise, it is of type FatalErrorException.
 * This is useful if you are sure
 * that all processes will perform the same operation. In this case,
 * this allows only one error message to be generated and the code to stop
 * cleanly.
 *
 * It is also possible to retrieve a singleton instance of a service,
 * via getSingleton(). The available singleton instances
 * are referenced in the code configuration file (see \ref arcanedoc_core_types_codeconfig).
 */
template<typename InterfaceType>
class ServiceBuilder
{
 public:

  //! Instantiation to create a subdomain service.
  ServiceBuilder(ISubDomain* sd)
  : m_service_finder(sd->application(),ServiceBuildInfoBase(sd))
  {}
  //! Instantiation to create a subdomain service associated with the \a mesh_handle
  ServiceBuilder(const MeshHandle& mesh_handle)
  : m_service_finder(mesh_handle.application(),ServiceBuildInfoBase(mesh_handle))
  {}
  //! Instantiation to create a session service.
  ServiceBuilder(ISession* session)
  : m_service_finder(session->application(),ServiceBuildInfoBase(session))
  {}
  //! Instantiation to create an application service.
  ServiceBuilder(IApplication* app)
  : m_service_finder(app,ServiceBuildInfoBase(app))
  {}
  //! Instantiation to create a dataset options service
  ServiceBuilder(IApplication* app,ICaseOptions* opt)
  : m_service_finder(app,ServiceBuildInfoBase(_arcaneDeprecatedGetSubDomain(opt),opt))
  {}
  
  ~ServiceBuilder(){ }

 public:
  
  /*!
   * \brief Creates an instance implementing the \a InterfaceType interface.
   *
   * The instance is created using the factory registered under the name \a name.
   *
   * By default, an exception is thrown if the specified service is not found.
   * It is possible to change this behavior by specifying SB_AllowNull in \a properties,
   * in which case the function returns a null pointer if the specified service does not exist.
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
   * \brief Creates an instance implementing the \a InterfaceType interface.
   *
   * The instance is created using the factory registered under the name \a name.
   * The returned pointer must be deallocated using delete.
   *
   * It is possible to specify the \a mesh on which the service will reside.
   * This is only useful for subdomain services. For session or application services,
   * this argument is not used.
   *
   * By default, an exception is thrown if the specified service is not found.
   * It is possible to change this behavior by specifying SB_AllowNull in \a properties,
   * in which case the function returns a null pointer if the specified service does not exist.
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
   * \brief Creates an instance of every service that implements \a InterfaceType.
   *
   * The created instances are stored in \a instances. The caller must
   * destroy them using the delete operator once they are no longer useful.
   */
  UniqueArray<Ref<InterfaceType>> createAllInstances()
  {
    return m_service_finder.createAll();
  }

  /*!
   * \brief Singleton instance of the service implementing the \a InterfaceType interface.
   *
   * The returned instance must not be destroyed.
   *
   * By default, an exception is thrown if the specified service is not found.
   * It is possible to change this behavior by specifying SB_AllowNull in \a properties,
   * in which case the function returns a null pointer if the specified service does not exist.
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
   * \brief Creates an instance implementing the \a InterfaceType interface.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   */
  static Ref<InterfaceType>
  createReference(ISubDomain* sd,const String& name,
                  eServiceBuilderProperties properties=SB_None)
  {
    return createReference(sd,name,0,properties);
  }

  /*!
   * \brief Creates an instance implementing the \a InterfaceType interface.
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
   * \brief Creates an instance implementing the \a InterfaceType interface.
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
   * \brief Creates an instance implementing the \a InterfaceType interface.
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

  //! Fills \a names with the names of services available to instantiate this interface
  void getServicesNames(Array<String>& names) const
  {
    m_service_finder.getServicesNames(names);
  }

 public:
  /*!
   * \brief Creates an instance of every service that implements \a InterfaceType.
   *
   * The created instances are stored in \a instances. The caller must
   * destroy them using the delete operator once they are no longer needed.
   *
   * \deprecated Use the overload that returns an array of references.
   */
  ARCCORE_DEPRECATED_2019("use createAllInstances(Array<Ref<InterfaceType>>) instead")
  void createAllInstances(Array<InterfaceType*>& instances)
  {
    m_service_finder.createAll(instances);
  }

  /*!
   * \brief Creates an instance implementing the interface \a InterfaceType.
   *
   * The instance is created using the factory registered under the name \a name.
   * The returned pointer must be deallocated using delete.
   *
   * By default, an exception is thrown if the specified service is not found.
   * It is possible to change this behavior by specifying SB_AllowNull in \a properties
   * in which case the function returns a null pointer if the specified service does not exist.
   *
   * \deprecated Use createReference() instead.
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
   * \brief Creates an instance implementing the interface \a InterfaceType.
   * \sa createInstance(const String& name,eServiceBuilderProperties properties)
   * \deprecated Use createReference() instead.
   */
  ARCCORE_DEPRECATED_2019("Use createReference() instead")
  static InterfaceType*
  createInstance(ISubDomain* sd,const String& name,
                 eServiceBuilderProperties properties=SB_None)
  {
    return createInstance(sd,name,0,properties);
  }

  /*!
   * \brief Creates an instance implementing the interface \a InterfaceType.
   *
   * The instance is created using the factory registered under the name \a name.
   * The returned pointer must be deallocated using delete.
   *
   * It is possible to specify the mesh \a mesh on which the service will rely.
   * This is only useful for subdomain services. For session
   * or application services, this argument is not used.
   *
   * By default, an exception is thrown if the specified service is not found.
   * It is possible to change this behavior by specifying SB_AllowNull in \a properties
   * in which case the function returns a null pointer if the specified service does not exist.
   *
   * \deprecated Use createReference() instead.
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
   * \brief Creates an instance implementing the interface \a InterfaceType.
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
   * \brief Creates an instance implementing the interface \a InterfaceType.
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
   * \brief Creates an instance implementing the interface \a InterfaceType.
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
    // No service available
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
/*!
 * \internal
 */
class ARCANE_CORE_EXPORT ServiceBuilderWithOptionsBase
{
 protected:

  ServiceBuilderWithOptionsBase(ICaseMng* cm)
  : m_case_mng(cm)
  {
  }

 protected:

  ReferenceCounter<ICaseOptions> _buildCaseOptions(const AxlOptionsBuilder::Document& options_doc) const;
  IApplication* _application() const;
  void _readOptions(ICaseOptions* opt) const;

 protected:

  ICaseMng* m_case_mng = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Utility class for instantiating a service of a given interface
 * with options.
 *
 * \warning Experimental API. Do not use outside of %Arcane
 */
template<typename InterfaceType>
class ServiceBuilderWithOptions
: private ServiceBuilderWithOptionsBase
{
 public:

  ServiceBuilderWithOptions(ICaseMng* cm) : ServiceBuilderWithOptionsBase(cm){}

 public:

  Ref<InterfaceType>
  createReference(const String& service_name,const AxlOptionsBuilder::Document& options_doc,
                  eServiceBuilderProperties properties=SB_None)
  {
    ReferenceCounter<ICaseOptions> opt(_buildCaseOptions(options_doc));
    ServiceBuilder<InterfaceType> sbi(_application(),opt.get());
    Ref<InterfaceType> s = sbi.createReference(service_name,properties);
    if (s.get()){
      _readOptions(opt.get());
    }
    return s;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
