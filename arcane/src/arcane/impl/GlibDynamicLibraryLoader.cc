// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GlibDynamicLibraryLoader.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Chargeur dynamique de bibliothèque avec Glib (utiliser gmodule).          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/IDynamicLibraryLoader.h"

#include "gmodule.h"

#include <iostream>
#include <set>
#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class GlibDynamicLibraryLoader;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT GlibDynamicLibrary
: public IDynamicLibrary
{
 public:
  GlibDynamicLibrary(GlibDynamicLibraryLoader* mng,GModule* gmodule)
  : m_manager(mng), m_gmodule(gmodule){}
 public:
  void close() override;
  void* getSymbolAddress(const String& symbol_name,bool* is_found) override;
 private:
  GlibDynamicLibraryLoader* m_manager;
  GModule* m_gmodule;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une chargeur dynamique de bibliothèque.
 */
class ARCANE_IMPL_EXPORT GlibDynamicLibraryLoader
: public IDynamicLibraryLoader
{
 public:
   GlibDynamicLibraryLoader() : m_is_verbose(false){}
   virtual ~GlibDynamicLibraryLoader(){} //!< Libère les ressources

 public:

  void build() override
  {
    String s = platform::getEnvironmentVariable("ARCANE_VERBOSE_DYNAMICLIBRARY");
    if (s=="1" || s=="true")
      m_is_verbose = true;
  }

  IDynamicLibrary* open(const String& directory,const String& name) override
  {
    IDynamicLibrary* dl = _tryOpen(directory, name);
    if (!dl){
      // Si on ne trouve pas, essaie avec l'extension '.dll' ou '.dylib' car sous
      // windows ou macos, certaines version de la GLIB prefixent automatiquement le
      // nom de la bibliothèque par 'lib' si elle ne finit pas par '.dll' ou '.dylib'.
#ifdef ARCANE_OS_WINDOWS
      dl = _tryOpen(directory, name + ".dll");
#endif
#ifdef ARCANE_OS_MACOS
      dl = _tryOpen(directory, "lib" + name + ".dylib");
#endif
    }
	if (!dl){
      // Si on ne trouve pas, essaie en cherchant à côté du binaire
      dl = _tryOpen(".", name);
    }
    if (!dl){
      // Si on ne trouve pas, essaie en cherchant à côté du binaire
      // et avec l'extension dll ou dylib
#ifdef ARCANE_OS_WINDOWS
      dl = _tryOpen(".", name + ".dll");
#endif
#ifdef ARCANE_OS_MACOS
      dl = _tryOpen(".", "lib" + name + ".dylib");
#endif
    }
    return dl;
  }

  IDynamicLibrary* _tryOpen(const String& directory, const String& name)
  {
    const gchar* gdirectory = reinterpret_cast<const gchar*>(directory.utf8().data());
    const gchar* gname = reinterpret_cast<const gchar*>(name.utf8().data());
    gchar* full_path = g_module_build_path(gdirectory, gname);
    if (m_is_verbose) {
      std::cout << "** Load Dynamic Library '" << full_path << "'...";
    }
    GModule* gmodule = g_module_open(full_path, GModuleFlags());
    g_free(full_path);
    if (m_is_verbose) {
      if (!gmodule){
        std::cout << " NOT FOUND\n";
      } else {
        std::cout << " OK\n";
      }
    }
    if (!gmodule)
      return nullptr;
    auto lib = new GlibDynamicLibrary(this,gmodule);
    m_opened_libraries.insert(lib);
    return lib;
  }
  
  void closeLibraries() override
  {
    // Cette méthode va modifier m opened libraries donc il faut le copier avant.
    std::vector<GlibDynamicLibrary*> libs(m_opened_libraries.begin(),m_opened_libraries.end());
    for( auto lib : libs ){
      lib->close();
      delete lib;
    }
  }

  void removeInstance(GlibDynamicLibrary* lib)
  {
    auto iter = m_opened_libraries.find(lib);
    if (iter!=m_opened_libraries.end())
      m_opened_libraries.erase(iter);
  }

 private:

  bool m_is_verbose;
  std::set<GlibDynamicLibrary*> m_opened_libraries;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GlibDynamicLibrary::
close()
{
  if (!m_gmodule)
    return;
  bool is_ok = g_module_close(m_gmodule);
  m_gmodule = 0;
  if (!is_ok)
    std::cerr << "WARNING: can not unload module\n";
  m_manager->removeInstance(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* GlibDynamicLibrary::
getSymbolAddress(const String& symbol_name,bool* is_found)
{
  if (is_found)
    *is_found = false;
  if (!m_gmodule)
    return nullptr;
  const gchar* gname = reinterpret_cast<const gchar*>(symbol_name.utf8().data());
  void* symbol_addr = nullptr;
  bool r = ::g_module_symbol(m_gmodule,gname,&symbol_addr);
  if (is_found)
    (*is_found) = r;
  return symbol_addr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IDynamicLibraryLoader*
createGlibDynamicLibraryLoader()
{
  IDynamicLibraryLoader* idll = new GlibDynamicLibraryLoader();
  idll->build();
  return idll;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

