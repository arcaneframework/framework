// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NativeDynamicLibraryLoader.cc                               (C) 2000-2026 */
/*                                                                           */
/* Dynamic library loader with Glib (uses gmodule).                          */
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/internal/IDynamicLibraryLoader.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

#ifdef ARCCORE_OS_WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::NativeLoading
{
class NativeLibrary;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_OS_WIN32
using NativeHandle = HMODULE;
#else
using NativeHandle = void*;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::string
_toStdString(const String& value)
{
  return std::string(value.toStdStringView());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::string
_fileName(const std::string& name)
{
#ifdef ARCCORE_OS_WIN32
  if (name.size() >= 4 && name.substr(name.size() - 4) == ".dll")
    return name;
  return name + ".dll";
#elif defined(ARCCORE_OS_MACOS)
  std::string result = name;
  if (result.rfind("lib", 0) != 0)
    result = "lib" + result;
  if (result.size() < 6 || result.substr(result.size() - 6) != ".dylib")
    result += ".dylib";
  return result;
#else
  std::string result = name;
  if (result.rfind("lib", 0) != 0)
    result = "lib" + result;
  if (result.size() < 3 || result.substr(result.size() - 3) != ".so")
    result += ".so";
  return result;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::string
_path(const std::string& directory, const std::string& file)
{
  if (directory.empty() || directory == ".")
    return "./" + file;
  if (directory.back() == '/' || directory.back() == '\\')
    return directory + file;
#ifdef ARCCORE_OS_WIN32
  return directory + "\\" + file;
#else
  return directory + "/" + file;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_OS_WIN32
std::wstring
_toWideString(const std::string& value)
{
  int n = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                static_cast<int>(value.size()), nullptr, 0);
  if (n <= 0)
    return {};
  std::wstring result(static_cast<size_t>(n), L'\0');
  if (!::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                             static_cast<int>(value.size()), result.data(), n))
    return {};
  return result;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NativeLoader
: public IDynamicLibraryLoader
{
 public:

  NativeLoader()
  {
    String value = platform::getEnvironmentVariable("ARCANE_VERBOSE_DYNAMICLIBRARY");
    m_is_verbose = (value == "1" || value == "true");
  }

  void build() override {}

  IDynamicLibrary* open(const String& directory, const String& name) override;
  void closeLibraries() override;
  void removeInstance(NativeLibrary* library);

 private:

  IDynamicLibrary* _tryOpen(const std::string& path);

 private:

  bool m_is_verbose = false;
  std::set<NativeLibrary*> m_opened_libraries;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NativeLibrary
: public IDynamicLibrary
{
 public:

  NativeLibrary(NativeLoader* manager, NativeHandle handle)
  : m_manager(manager)
  , m_handle(handle)
  {}
  void close() override;
  void* getSymbolAddress(const String& symbol_name, bool* is_found) override;

 private:

  NativeLoader* m_manager;
  NativeHandle m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDynamicLibrary* NativeLoader::
_tryOpen(const std::string& path)
{
  if (m_is_verbose)
    std::cout << "** Load Dynamic Library '" << path << "'...";
#ifdef ARCCORE_OS_WIN32
  std::wstring wide_path = _toWideString(path);
  NativeHandle handle = wide_path.empty() ? nullptr : ::LoadLibraryW(wide_path.c_str());
#else
  NativeHandle handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (m_is_verbose)
    std::cout << (handle ? " OK\n" : " NOT FOUND\n");
  if (!handle)
    return nullptr;
  auto* library = new NativeLibrary(this, handle);
  m_opened_libraries.insert(library);
  return library;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDynamicLibrary* NativeLoader::
open(const String& directory, const String& name)
{
  std::string requested_name = _toStdString(name);
  std::string built_name = _fileName(requested_name);
  std::vector<std::string> names{ built_name };
  if (built_name != requested_name)
    names.push_back(requested_name);
  std::string requested_directory = _toStdString(directory);
  std::vector<std::string> directories{ requested_directory };
  if (requested_directory != ".")
    directories.emplace_back(".");
  for (const auto& dir : directories)
    for (const auto& file : names)
      if (auto* library = _tryOpen(_path(dir, file)))
        return library;
  return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NativeLoader::
closeLibraries()
{
  std::vector<NativeLibrary*> libraries(m_opened_libraries.begin(), m_opened_libraries.end());
  for (auto* library : libraries) {
    library->close();
    delete library;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NativeLoader::
removeInstance(NativeLibrary* library)
{
  m_opened_libraries.erase(library);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NativeLibrary::
close()
{
  if (!m_handle)
    return;
#ifdef ARCCORE_OS_WIN32
  bool is_ok = ::FreeLibrary(m_handle) != 0;
#else
  bool is_ok = ::dlclose(m_handle) == 0;
#endif
  m_handle = nullptr;
  if (!is_ok)
    std::cerr << "WARNING: can not unload module\n";
  m_manager->removeInstance(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* NativeLibrary::
getSymbolAddress(const String& symbol_name, bool* is_found)
{
  if (is_found)
    *is_found = false;
  if (!m_handle)
    return nullptr;
  std::string symbol = _toStdString(symbol_name);
#ifdef ARCCORE_OS_WIN32
  void* address = reinterpret_cast<void*>(::GetProcAddress(m_handle, symbol.c_str()));
  bool found = (address != nullptr);
#else
  (void)::dlerror();
  void* address = ::dlsym(m_handle, symbol.c_str());
  bool found = (::dlerror() == nullptr);
#endif
  if (is_found)
    *is_found = found;
  return address;
}

namespace
{
  NativeLoader native_loader;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::NativeLoading

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

extern "C++" IDynamicLibraryLoader*
getDefaultGlibDynamicLibraryLoader();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDynamicLibraryLoader* IDynamicLibraryLoader::
getDefault()
{
  // At the moment use Glib dynamic library loader by default
  bool use_glib = true;
  String value = platform::getEnvironmentVariable("ARCANE_USE_GLIB_DYNAMICLIBRARY_LOADER");
  if (value == "0")
    use_glib = false;
#if !defined(ARCCORE_HAS_GLIB)
  use_glib = false;
#endif
  if (use_glib)
    return getDefaultGlibDynamicLibraryLoader();
  return &NativeLoading::native_loader;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
