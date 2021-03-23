// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_APPLI_IAPPSERVICEMNG_H
#define ARCGEOSIM_APPLI_IAPPSERVICEMNG_H

/**
 * \author Jean-Marc GRATIEN
 * \version 1.0
 * \brief Interface permettant de definir les services uniques d'une application
 * 
 * Exemple d'utilisation :
 * \code
 * IAppServiceMng* app_service_mng = IAppServiceMng::instance(subDomain()->serviceMng());
 * IGeometryMng * service = app_service_mng->find<IGeometryMng>(true) ;
 * \endcode
 */

#include <arcane/utils/ListImpl.h>
#include <arcane/utils/String.h>
#include <arcane/IService.h>
#include <typeinfo>

#include <map> 

namespace Arcane {
  class ITimeLoopMng;
  class IServiceMng;
  class ITraceMng;
}

using namespace Arcane;


class IAppServiceMng
{
public:
  //! Constructeur de la classe
  IAppServiceMng();
  
  //! Destructeur de la classe
  virtual ~IAppServiceMng() {}
  
protected:
  //! Initialisation  
  virtual void initializeAppServiceMng() = 0;
  
public:
  //! Ajout d'un service 
  void addService(IService* service);
  
  //! Ajout d'un service 
  void addService(IService* service, const String & type_name);

  //! Information sur un service absent
  void addMissingServiceInfo(const String & type_id, const String & type_name);

  //! Recherche d'un service
  /*! @param delegate_error : true si la gestion d'erreur est déléguée (ie si non trouvé)
   *  L'initialisation du service trouvé est à la charge de IAppServiceMng
   */
  template<class T>
  T* find(bool delegate_error)
  {
    if (delegate_error)
      _checkBeforeFind(typeid(T*).name());
    for( typename ListImplT< std::pair<IService*,bool> >::iterator iter=m_services.begin(); iter!=m_services.end(); ++iter)
      {
        IService * s = (*iter).first;
        {
          T* m = dynamic_cast<T*>(s);
          if (m) 
            {
              if((*iter).second) // true si pas encore initialisé
              {
                m->init();
                (*iter).second = false ; // mémorisation de l'initialisation
              }
              return m ;
            }
        }
      }

    if (delegate_error)
      _checkNotFound(typeid(T).name());

    return NULL ;
  }

  //! Définit le gestion de boucle pour des messages d'erreur plus précis
  void setTimeLoopMng(ITimeLoopMng * time_loop_mng);

  //! Accès à l'instance de IAppServiceMng
  static IAppServiceMng * instance(IServiceMng * sm = NULL);

private:
  void _checkBeforeFind(const char * name);
  void _checkNotFound(const char * name);
  ITraceMng * traceMng();
  
private:
  ListImplT< std::pair<IService *,bool> > m_services ; //!< liste des services avec flag d'initialisation (true => non initialisé)
  std::map<String,String> m_missing_service_infos; //!< Information descriptive des services manquant
  ITimeLoopMng * m_time_loop_mng; //!< Lien au gestionnaire de boucle en temps
  static IAppServiceMng * m_instance;
};

#endif /* ARCGEOSIM_APPLI_IAPPSERVICEMNG_H */

