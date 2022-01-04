// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Configuration.cc                                            (C) 2000-2014 */
/*                                                                           */
/* Gestion des options de configuration de l'exécution.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/Configuration.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IApplication;
class Configuration;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConfigurationMng
: public TraceAccessor
, public IConfigurationMng
{
 public:

  ConfigurationMng(ITraceMng* tm);
  ~ConfigurationMng();

 public:
  
  virtual IConfiguration* defaultConfiguration() const;
  virtual IConfiguration* createConfiguration();

 private:

  Configuration* m_default_configuration;

 private:
  
  Configuration* _createConfiguration();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConfigurationSection
: public IConfigurationSection
{
 public:

  ConfigurationSection(const Configuration* configuration,const String& base_name)
  : m_configuration(configuration), m_base_name(base_name){}

  virtual ~ConfigurationSection() {}

 public:

  virtual Int32 value(const String& name,Int32 default_value) const;
  virtual Int64 value(const String& name,Int64 default_value) const;
  virtual Real value(const String& name,Real default_value) const;
  virtual bool value(const String& name,bool default_value) const;
  virtual String value(const String& name,const String& default_value) const;
  virtual String value(const String& name,const char* default_value) const;

  virtual Integer valueAsInteger(const String& name,Integer default_value) const
  { return value(name,default_value); }
  virtual Int32 valueAsInt32(const String& name,Int32 default_value) const
  { return value(name,default_value); }
  virtual Int64 valueAsInt64(const String& name,Int64 default_value) const 
  { return value(name,default_value); }
  virtual Real valueAsReal(const String& name,Real default_value) const
  { return value(name,default_value); }
  virtual bool valueAsBool(const String& name,bool default_value) const
  { return value(name,default_value); }
  virtual String valueAsString(const String& name,const String& default_value) const
  { return value(name,default_value); }

 private:

  const Configuration* m_configuration;
  String m_base_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Configuration
: public TraceAccessor
, public IConfiguration
{
  struct ConfigValue
  {
    ConfigValue(const String& value,Integer priority)
    : m_value(value), m_priority(priority){}
    const String& value() const { return m_value; }
    Integer priority() const { return m_priority; }
   private:
    String m_value;
    Integer m_priority;
  };

  typedef std::map<String,ConfigValue> KeyValueMap;

 public:

  Configuration(ConfigurationMng* cm,ITraceMng* tm);

 public:

  IConfigurationSection* createSection(const String& name) const override
  {
    return new ConfigurationSection(this,name);
  }

  IConfigurationSection* mainSection() const override { return m_main_section.get(); }

  void addValue(const String& name,const String& value,Integer priority) override;
  IConfiguration* clone() const override;
  void merge(const IConfiguration* c) override;
  void dump() const override;
  void dump(ostream& o) const override;

 public:

  template<typename T> T getValue(const String& base_name,const String& name,T default_value) const
  {
    T v = T();
    KeyValueMap::const_iterator i;
    if (base_name.null())
      i = m_values.find(name);
    else
      i = m_values.find(base_name+"."+name);

    if (i==m_values.end())
      return default_value;

    String value = i->second.value();
    if (builtInGetValue(v,value))
      throw FatalErrorException(A_FUNCINFO,String::format("Can not convert '{0}' to type '{1}'",value,_typeName((T*)0)));

    return v;
  }

 private:

  static const char* _typeName(Int32*) { return "Int32"; }
  static const char* _typeName(Int64*) { return "Int64"; }
  static const char* _typeName(Real*) { return "Real"; }
  static const char* _typeName(bool*) { return "bool"; }
  static const char* _typeName(String*) { return "String"; }
  
  void _checkAdd(const String& name,const String& value,Integer priority)
  {
    KeyValueMap::iterator i = m_values.find(name);
    //info() << "CHECK_ADD name=" << name << " value=" << value << " priority=" << priority;
    if (i==m_values.end()){
      //info() << "NOT_FOUND name=" << name << " value=" << value << " priority=" << priority;
      m_values.insert(std::make_pair(name,ConfigValue(value,priority)));
    }
    else{
      Integer orig_priority = i->second.priority();
      //info() << "FOUND name=" << name << " value=" << value << " orig_priority=" << orig_priority;
      if (priority<orig_priority){
        //info() << "REPLACING name=" << name << " value=" << value << " new_priority=" << priority;
        i->second = ConfigValue(value,priority);
      }
    }
  }

 private:

  ConfigurationMng* m_configuration_mng;
  KeyValueMap m_values;
  ScopedPtrT<IConfigurationSection> m_main_section;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ConfigurationSection::
value(const String& name,Int32 default_value) const
{
  return m_configuration->getValue(m_base_name,name,default_value);
}

Int64 ConfigurationSection::
value(const String& name,Int64 default_value) const
{
  return m_configuration->getValue(m_base_name,name,default_value);
}

Real ConfigurationSection::
value(const String& name,Real default_value) const
{
  return m_configuration->getValue(m_base_name,name,default_value);
}

bool ConfigurationSection::
value(const String& name,bool default_value) const
{
  return m_configuration->getValue(m_base_name,name,default_value);
}

String ConfigurationSection::
value(const String& name,const String& default_value) const
{
  return m_configuration->getValue(m_base_name,name,default_value);
}

String ConfigurationSection::
value(const String& name,const char* default_value) const
{
  return m_configuration->getValue(m_base_name,name,String(default_value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Configuration::
Configuration(ConfigurationMng* cm,ITraceMng* tm)
: TraceAccessor(tm),
  m_configuration_mng(cm),
  m_main_section(new ConfigurationSection(this,String()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IConfiguration* Configuration::
clone() const
{
  Configuration* cf = new Configuration(m_configuration_mng,traceMng());

  for( auto& i : m_values ){
    cf->m_values.insert(std::make_pair(i.first,i.second));
  }

  return cf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Configuration::
dump() const
{
  OStringStream ostr;
  dump(ostr());
  info() << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Configuration::
dump(ostream& o) const
{
  o << "Configuration:\n";
  for( auto& i : m_values ){
    String s1 = i.first;
    String s2 = i.second.value();
    o << " name=" << s1 << " value=" << s2 << " (" << i.second.priority() << ")\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Configuration::
merge(const IConfiguration* c)
{
  const Configuration* cc = ARCANE_CHECK_POINTER(dynamic_cast<const Configuration*>(c));
  for( auto& i : cc->m_values ){
    String s1 = i.first;
    const ConfigValue& cv = i.second;
    //String s2 = i->second.value();
    _checkAdd(s1,cv.value(),cv.priority());
    //info() << "MERGING CONFIGURATION name=" << s1 << " value=" << s2;
    //m_values[s1] = s2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Configuration::
addValue(const String& name,const String& value,Integer priority)
{
  _checkAdd(name,value,priority);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConfigurationMng::
ConfigurationMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_default_configuration(0)
{
  m_default_configuration = _createConfiguration();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConfigurationMng::
~ConfigurationMng()
{
  delete m_default_configuration;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Configuration* ConfigurationMng::
_createConfiguration()
{
  return new Configuration(this,traceMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IConfiguration* ConfigurationMng::
defaultConfiguration() const
{
  return m_default_configuration;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IConfiguration* ConfigurationMng::
createConfiguration()
{
  return _createConfiguration();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IConfigurationMng*
arcaneCreateConfigurationMng(ITraceMng* tm)
{
  IConfigurationMng *cm = new ConfigurationMng(tm);
  return cm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
