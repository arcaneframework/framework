// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Utils/Utils.h"
#include "Appli/IAppServiceMng.h"

#include <arcane/IService.h>
#include <arcane/IModule.h>
#include <arcane/IServiceMng.h>
#include <arcane/IBase.h>
#include <arcane/IModule.h>
#include <arcane/IEntryPoint.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/ServiceFinder.h>
#include <arcane/utils/ITraceMng.h>
#include <arcane/utils/FatalErrorException.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAppServiceMng * IAppServiceMng::m_instance = NULL;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAppServiceMng::
IAppServiceMng() 
  : m_time_loop_mng(NULL)
{
  if (m_instance)
    throw FatalErrorException("IAppServiceMng already registered");
  m_instance = this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
IAppServiceMng::
addService(IService* service)
{
  traceMng()->info() << "Adding Service " << typeid(*service).name() << " (" << service << ")";
  m_services.add(std::pair<IService*,bool>(service,true)) ;
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
IAppServiceMng::
addService(IService* service, const String & type_name)
{
  traceMng()->info() << "Adding Service " << typeid(*service).name() << " for " << type_name << " (" << service << ")";
  m_services.add(std::pair<IService*,bool>(service,true)) ;
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void 
IAppServiceMng::
addMissingServiceInfo(const String & type_id, const String & type_name)
{
  traceMng()->info() << "Adding missing Service " << type_id << " for " << type_name;
  m_missing_service_infos[type_id] = type_name ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
IAppServiceMng::
setTimeLoopMng(ITimeLoopMng * time_loop_mng)
{
  m_time_loop_mng = time_loop_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
IAppServiceMng::
_checkBeforeFind(const char * name)
{
  std::map<String,String>::const_iterator i_missing_service = m_missing_service_infos.find(name);
  if (i_missing_service != m_missing_service_infos.end())
    {
      String context;
      if (m_time_loop_mng) {
        IEntryPoint * entry = m_time_loop_mng->currentEntryPoint();
        context = " in " + entry->module()->name() + "::" + entry->name();
      }
      traceMng()->fatal() 
        << "Requested optional " << i_missing_service->second << " service" << context << " not configured in your .arc file";
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
IAppServiceMng::
_checkNotFound(const char * name)
{
  String context;
  if (m_time_loop_mng) {
    IEntryPoint * entry = m_time_loop_mng->currentEntryPoint();
    context = " in " + entry->module()->name() + "::" + entry->name();
  }
  traceMng()->fatal()
    << "Requested service typeid " << name << context << " cannot be found";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng *
IAppServiceMng::
traceMng()
{
  IService * s;
  IModule * m;
  if ((s = dynamic_cast<IService*>(this))) { // IAppServiceMng is a service
    return s->serviceParent()->traceMng();
  } else if ((m = dynamic_cast<IModule*>(this))) {
    return m->traceMng();
  } else {
    throw FatalErrorException("IAppServiceMng is neither a IService nor a IModule");
    return NULL;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IAppServiceMng *
IAppServiceMng::
instance(IServiceMng * sm)
{
  if (m_instance) 
    {
      m_instance->initializeAppServiceMng();
      return m_instance;
    }
  else if (sm == NULL)
    {
      throw FatalErrorException("Cannot find IAppServiceMng by static registration");
      return NULL;
    }
  else 
    {
      IAppServiceMng * app_service_mng = ServiceFinderT<IAppServiceMng>(sm).findSingleton();
      if (app_service_mng != NULL)
        {
          app_service_mng->initializeAppServiceMng();
        }
      else
        {
          throw FatalErrorException("Cannot find IAppServiceMng by ServiceFinder");
        }
    return app_service_mng;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
