/*---------------------------------------------------------------------------*/
/* Stat.h                                                      (C) 2000-2018 */
/*                                                                           */
/* Statistiques sur le parallélisme.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_STAT_H
#define ARCCORE_MESSAGEPASSING_STAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/IStat.h"

#include "arccore/base/String.h"

#include <map>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Statistiques sur le parallélisme.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Stat
: public IStat
{
 public:

 public:

  typedef std::map<String,OneStat*> OneStatMap;
  typedef std::pair<String,OneStat*> OneStatValue;

 public:

  Stat();
  //! Libère les ressources.
  virtual ~Stat();

 public:

 public:

  void add(const String& name,double elapsed_time,Int64 msg_size) override;
  void enable(bool is_enabled) override { m_is_enabled = is_enabled; }

  void print(std::ostream& o);

  const OneStatMap& stats() const { return m_list; }

 private:

  bool m_is_enabled;
  OneStatMap m_list;

 private:

  OneStat* _find(const String& name)
  {
    OneStatMap::const_iterator i = m_list.find(name);
    if (i!=m_list.end()){
      return i->second;
    }
    OneStat* os = new OneStat(name);
    // Important: utiliser os.m_name car m_list stocke juste un
    // pointeur sur la chaîne de caractère.
    m_list.insert(OneStatValue(os->name(),os));
    return os;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
