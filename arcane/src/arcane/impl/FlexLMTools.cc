// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FlexLMTools.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Implémentation d'une interface pour les outils FlexLM.                    */
/*---------------------------------------------------------------------------*/

#include "arcane/impl/FlexLMTools.h"

/* L'interface à FlexNet ici employée est une surcouche à FlexNet de TAD 
 * prenant en charge par exemple le rappel périodique du contrôle de licence.
 * Le recheck par défaut est tous les 120s (cf setCheckInterval)
 */

#define FLEXLMAPI_IS_STATIC_LIBRARY
#define c_plusplus
#include <FlexlmAPI.h>

#include "arcane/IParallelSuperMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LicenseErrorException::
LicenseErrorException(const String& where)
: Exception("LicenseError",where)
{
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LicenseErrorException::
LicenseErrorException(const String& where,const String& message)
: Exception("LicenseError",where,message)
{
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LicenseErrorException::
LicenseErrorException(const TraceInfo& where)
: Exception("LicenseError",where)
{
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LicenseErrorException::
LicenseErrorException(const TraceInfo& where,const String& message)
: Exception("LicenseError",where,message)
{
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LicenseErrorException::
explain(std::ostream& m) const
{
	m << "Licensing error occured.\n"
    << "Excution stopped.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LicenseErrorException::
write(std::ostream& o) const
{
  o << name() << "\n";
#ifdef ARCANE_DEBUG
  o << "Exception thrown in: '" << where() << "\n";
#endif /* ARCANE_DEBUG */
  if (!message().null())
    o << "Message: " << message() << '\n';
  this->explain(o);
#ifdef ARCANE_DEBUG
  String st = stackTrace().toString();
  if (!st.null()){
    o << "\nCall stack:\n";
    o << st << '\n';
  }
#endif /* ARCANE_DEBUG */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FlexLMMng* FlexLMMng::m_instance = NULL ;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FlexLMMng::
FlexLMMng()
  : m_parallel_super_mng(NULL)
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FlexLMMng*
FlexLMMng::
instance()
{
  if(m_instance==NULL)
    m_instance = new FlexLMMng() ;
  return m_instance ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
FlexLMMng::
init(IParallelSuperMng * parallel_super_mng)
{
  if (m_parallel_super_mng != NULL)
    throw LicenseErrorException(A_FUNCINFO,"FlexLMMng already initialized");

  // Test en mode master check le droit de décentralisé le contrôle
  m_is_master = (parallel_super_mng->commRank() == 0);

  // Marque l'initialisation effective; on peut maintenant utiliser le système de licence
  m_parallel_super_mng = parallel_super_mng;

  FlexLMTools<ArcaneFeatureModel> license_tool;
#ifndef ARCANE_TEST_RLM
  license_tool.checkLicense(ArcaneFeatureModel::ArcaneCore,true); // do_fatal=true
#else
  license_tool.checkLicense(ArcaneFeatureModel::Arcane,true); // do_fatal=true
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
FlexLMMng::
setCheckInterval(const Integer t)
{
  // Toutes les spécificités sont gérées dans la fonction
  setcheckinterval(t);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
FlexLMMng::
checkLicense(const String name, const Real version, const bool do_fatal) const
{
  if (m_parallel_super_mng == NULL)
    throw LicenseErrorException(A_FUNCINFO,"FlexLMMng not initialized");

  Integer test = 0;
  if (m_is_master)
    test = license_test((char*)name.localstr(),(char*)String::format("{0}",version).localstr());
  m_parallel_super_mng->broadcast(IntegerArrayView(1,&test),0);

  if (test != 0 && do_fatal)
    throw LicenseErrorException(A_FUNCINFO,String::format("Checking feature {0} (v{1}) has failed\nFeature info: {2}",name,version,featureInfo(name,version)));
  return (test == 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
FlexLMMng::
getLicense(const String name, const Real version, Integer nb_licenses)
{
  if (m_parallel_super_mng == NULL)
    throw LicenseErrorException(A_FUNCINFO,"FlexLMMng not initialized");

  Integer error = 0;
  if (m_is_master)
    {
      m_features[name] += nb_licenses ;
      for(Integer i=0;i<nb_licenses;++i)
        error += license_begin_rc((char*)name.localstr(), (char*)String::format("{0}",version).localstr()) ;
    }
  m_parallel_super_mng->broadcast(IntegerArrayView(1,&error),0);
  if (error != 0)
    throw LicenseErrorException(A_FUNCINFO,String::format("Cannot checkout {0} license{1} for feature {2} (v{3})\nFeature info: {4}",nb_licenses,((nb_licenses>1)?"s":""),name,version,featureInfo(name,version)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
FlexLMMng::
releaseLicense(const String name, Integer nb_licenses)
{
  if (m_parallel_super_mng == NULL)
    throw LicenseErrorException(A_FUNCINFO,"FlexLMMng not initialized");

  if (nb_licenses == 0) 
    return;

  if (m_is_master) 
    {
      FeatureMapType::iterator finder = m_features.find(name);
      
      // Pas de code d'erreur pour compatibilité avec la précédente version
      if (finder == m_features.end()) return;
      
      Integer & nb_allocated_licenses = finder->second;
      if (nb_licenses == 0) nb_licenses = nb_allocated_licenses;
      for(Integer i=0;i<nb_licenses;++i)
        license_end_no_quit((char*)name.localstr()) ;
      
      nb_allocated_licenses -= nb_licenses;
      if (nb_allocated_licenses < 0) nb_allocated_licenses = 0;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
FlexLMMng::
releaseAllLicenses()
{
  if (m_parallel_super_mng == NULL)
    return;

  if (m_is_master)
    {
      for(FeatureMapType::iterator iter = m_features.begin(); iter!=m_features.end(); ++iter)
        {
          const String name = (*iter).first ;
          Integer nb_licenses = (*iter).second ;
          for(Integer i=0; i<nb_licenses; ++i)
            license_end_no_quit((char*) name.localstr()) ;
          (*iter).second = 0 ;
        }
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String 
FlexLMMng::
featureInfo(const String name, const Real version) const
{
  String info;
  if (m_is_master) 
    {
      StringBuilder info_builder;
      if (license_test((char*)name.localstr(),(char*)String::format("{0}",version).localstr()) == 0)
        {
          Integer count = 0;
          char ** users = get_user_list((char*)name.localstr());
          while(users[count]) { info_builder += String::format("{0} ",users[count++]); }
          if (count == 0) {
            info_builder = String::format("no declared user");
          } else {
            info_builder = String::format("Used by {0} user{1} : ",count,((count>1)?"s":""),info_builder.toString());
          }
          
          // license_test_expiration semble buggué : segfault
          // int expiration_code = license_test_expiration((char*)name.localstr());
          // info_builder += String::format("\nExpiration Code {0}",expiration_code);

          info = info_builder.toString();
        }
      else
        {
          info = "Unknown feature";
        }
    }

  // remplace m_parallel_super_mng->broadcastString(info,0) qui n'existe pas dans IParallelSuperMng
  Integer len_info[1] = { info.utf8().size() };
  m_parallel_super_mng->broadcast(IntegerArrayView(1,len_info),0);
  if (m_is_master) {
    ByteUniqueArray utf8_array(info.utf8());
    m_parallel_super_mng->broadcast(utf8_array,0);
  } else {
    ByteUniqueArray utf8_array(len_info[0]);
    m_parallel_super_mng->broadcast(utf8_array,0);
    info = String::fromUtf8(utf8_array);
  }

  return info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String ArcaneFeatureModel::m_arcane_feature_name[] = 
  {
#ifndef ARCANE_TEST_RLM
    "ArcaneCore",
#else
    "Arcane",
#endif
  };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
