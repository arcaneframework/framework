/*---------------------------------------------------------------------------*/
/* GlibAdapter.cc                                              (C) 2000-2018 */
/*                                                                           */
/* Classes utilitaires pour s'adapter aux différentes versions de la 'glib'. */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/GlibAdapter.h"

#include <glib.h>

// A partir de 2.32, 'glib' utilise un nouveau mécanisme pour gérer
// tout ce qui concerne le multi-threading. En particulier, toutes les
// fonctions de création/destruction changent.

// A terme, ces fonctionnalités seront disponibles dans la norme C++
// et lorsqu'on pourra utiliser des compilateurs récents elles ne devraient
// plus être utilisées.

#if GLIB_CHECK_VERSION(2,32,0)
#define ARCCORE_GLIB_HAS_NEW_THREAD
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief GMutex de la glib.
 */
class GlibMutex::Impl
{
 public:
  Impl() ARCCORE_NOEXCEPT
  : m_mutex(nullptr)
  {
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
    m_mutex = &m_mutex_instance;
    g_mutex_init(m_mutex);
#else
    m_mutex = g_mutex_new();
#endif
  }
  ~Impl()
  {
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
    g_mutex_clear(m_mutex);
#else
    g_mutex_free(m_mutex);
#endif
  }
 public:
  GMutex* value() const { return m_mutex; }
  void lock() { g_mutex_lock(m_mutex); }
  void unlock() { g_mutex_unlock(m_mutex); }
 private:
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
  GMutex m_mutex_instance;
#endif
  GMutex* m_mutex;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlibMutex::
GlibMutex() ARCCORE_NOEXCEPT
: m_p(new Impl())
{
}

GlibMutex::
~GlibMutex()
{
  delete m_p;
}

void GlibMutex::lock() { m_p->lock(); }
void GlibMutex::unlock() { m_p->unlock(); }

GlibMutex::Lock::Lock(GlibMutex& x) : m_mutex(x.m_p){ m_mutex->lock(); }
GlibMutex::Lock::~Lock() { m_mutex->unlock(); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
static GPrivate null_gprivate = G_PRIVATE_INIT(nullptr);
#endif

/*!
 * \internal
 * \brief GPrivate de la glib.
 */
class GlibPrivate::Impl
{
 public:
  Impl() ARCCORE_NOEXCEPT
  : m_private(nullptr)
  {
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
    m_private_instance = null_gprivate;
    m_private = &m_private_instance;
#endif
  }
  ~Impl()
  {
  }
  void create()
  {
#ifndef ARCCORE_GLIB_HAS_NEW_THREAD
    if (!m_private)
      m_private = g_private_new(nullptr);
#endif
  }
  void setValue(void* value)
  {
    g_private_set(m_private,value);
  }
  void* getValue()
  {
    return g_private_get(m_private);
  }
 private:
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
  GPrivate m_private_instance;
#endif
  GPrivate* m_private;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlibPrivate::
GlibPrivate()
: m_p(new Impl())
{
}

GlibPrivate::
~GlibPrivate()
{
  delete m_p;
}

void GlibPrivate::create() { m_p->create(); }
void GlibPrivate::setValue(void* value) { m_p->setValue(value); }
void* GlibPrivate::getValue() { return m_p->getValue(); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GlibCond::Impl
{
 public:
  Impl() : m_cond(nullptr)
  {
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
    m_cond = &m_cond_instance;
    g_cond_init(m_cond);
#else
    m_cond = g_cond_new();
#endif
  }
  ~Impl()
  {
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
    g_cond_clear(m_cond);
#else
    g_cond_free(m_cond);
#endif
  }
 public:
  void broadcast() { g_cond_broadcast(m_cond); }
  void wait(GlibMutex::Impl* mutex) { g_cond_wait(m_cond,mutex->value()); }
 private:
#ifdef ARCCORE_GLIB_HAS_NEW_THREAD
  GCond m_cond_instance;
#endif
  GCond* m_cond;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GlibCond::GlibCond() : m_p(new GlibCond::Impl()){}
GlibCond::~GlibCond() { delete m_p; }
void GlibCond::broadcast() { m_p->broadcast(); }
void GlibCond::wait(GlibMutex* mutex) { m_p->wait(mutex->m_p); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
