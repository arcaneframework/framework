// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef PROPERTYHOLDER_H_
#define PROPERTYHOLDER_H_

#include <map>
#include <arcane/utils/String.h>
#include <arcane/utils/ITraceMng.h>

/**
  * \author Jean-Marc GRATIEN
  * \version 1.0
  * \class PropertyHolder
  * \brief template<class Property,class ValueType> permettant de gérer 
  *  une liste de propriétés de type ValueType associé à l'enum Property::eProperty
  * 
  * 
  * 
  * 
  */
template<class Property,class ValueType>
class PropertyHolder
{
private:
  
  template<class T>
  bool _PrintableValue() const
  {
    return true;
  }
  
  template<class T*>
  bool _PrintableValue() const
  {
    return false;
  } 
  
public:
  
  PropertyHolder()
  {
    m_name = "Undefined" ;
  };
  
	virtual ~PropertyHolder(){};
  
  typedef typename Property::eProperty PropertyType ;
  typedef std::map<PropertyType,ValueType> PropertyList ;
  typedef typename PropertyList::iterator PropertyListIter ;
  typedef typename PropertyList::const_iterator PropertyListConstIter ;
  
  //! check if has one property
  bool hasProperty(PropertyType property) const
  {
    return (m_properties.find(property)!=m_properties.end()) ;
  }
  
  //! return the value of one property, defaultValue if the property does not exist
  ValueType getProperty(PropertyType property,
                        ValueType default_value) const
  {
    PropertyListConstIter iter = m_properties.find(property) ;
    if(iter!=m_properties.end())
      return (*iter).second ;
    else
      return default_value ;
  }
  
  //! return the value of one property and set if the property exists
  ValueType getProperty(PropertyType property,
                        bool* hasProperty,
                        ValueType default_value) const
  {
    PropertyListConstIter iter = m_properties.find(property) ;
    if(iter!=m_properties.end())
    {
      *hasProperty = true ;
      return (*iter).second ;
    }
    else
    {
      *hasProperty = false ;
      return default_value ;
    }
  }
  
  ValueType getProperty(PropertyType property,
                        bool* hasProperty) const
  {
    PropertyListConstIter iter = m_properties.find(property) ;
    if(iter!=m_properties.end())
    {
      *hasProperty = true ;
      return (*iter).second ;
    }
    else
    {
      *hasProperty = false ;
      return m_default_value ;
    }
  }
  
  //! return the name of the property holder
  String getName() const { return m_name ; }
  
  //! set the name of the property holder
  void setName(String name) { m_name = name ; }
  
  //! set the value of one property
  void setProperty(PropertyType property, ValueType val) 
  {
     m_properties[property] = val ;
  }
  
  //! set the defaultValue to return when one property does not exist
  void setDefaultValue(ValueType val) { m_default_value = val ; }

  //! Print properties
  void printInfo(ITraceMng * trace_mng) const
  {
    ARCANE_ASSERT((trace_mng),("Trace manager pointer null"));
    
    for(PropertyListConstIter iProperty=m_properties.begin(); iProperty!=m_properties.end(); ++iProperty)
      {
        PropertyType property_type = iProperty->first;
                
        if(_PrintableValue<ValueType>())
          {
            trace_mng->info() << " * Property name : " << Property::name(property_type) 
                              << ", value = " << iProperty->second;
          }
        else
          {
            trace_mng->info() << " * Property name : " << Property::name(property_type) 
                              << ", value type not printable";
          }
      }
  }
  
private :
  
  //! Property holder name
  String m_name ;
  
  //!Default Value for all properties
  ValueType m_default_value ;
  
  //!Properties contener
  PropertyList m_properties ;
};

#endif /*PROPERTYHOLDER_H_*/
