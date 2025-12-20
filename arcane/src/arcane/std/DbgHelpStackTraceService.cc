// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DbgHelpStackTraceService.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Service de trace des appels de fonctions utilisant 'DbgHelp'.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/ISymbolizerService.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/StringBuilder.h"

#include "arccore/base/internal/DependencyInjection.h"

#include "arcane/core/AbstractService.h"
#include "arcane/core/ServiceFactory.h"

#include <Windows.h>
#include <winnt.h>
#include <DbgHelp.h>
#include <string>
#include <iostream>

// TODO: protéger les appels à Sym* car les méthodes ne sont pas thread-safe.
// TODO: ne pas retourner le chemin complet des fichiers mais uniquement les
// 3 ou 4 derniers.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_STD_EXPORT DbgHelpSymContainer
{
public:
  void init()
  {
    _init();
  }

  FixedStackFrameArray getStackFrames(int first_function)
  {
    FixedStackFrameArray frames;
    _init();
    if (!m_is_good)
      return frames;
    _getStack(first_function,frames);
    return frames;
  }
  String getStackSymbols(ConstArrayView<StackFrame> frames)
  {
    return _getStackSymbols(frames);
  }

 private:
  void _init()
  {
    if (m_is_init)
      return;
    m_is_init = true;
    ::SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_INCLUDE_32BIT_MODULES | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    if (!::SymInitialize(::GetCurrentProcess(), nullptr, true)) {
      m_is_good = false;
      return;
    }
    m_is_good = true;
  }
  void _getStack(int first_function, FixedStackFrameArray& frames);
  String _getStackSymbols(ConstArrayView<StackFrame> frames);
private:
  bool m_is_init = false;
  bool m_is_good = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DbgHelpSymContainer::
_getStack(int first_function, FixedStackFrameArray& frames)
{
  // Ajoute (+2) pour prendre en compte l'appel à cette méthode et celle du dessus.
  PVOID addrs[FixedStackFrameArray::MAX_FRAME] = { 0 };
  USHORT nb_frame = CaptureStackBackTrace(first_function + 2, FixedStackFrameArray::MAX_FRAME, addrs, NULL);
  for (USHORT i = 0; i < nb_frame; i++) {
    frames.addFrame(StackFrame((intptr_t)addrs[i]));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String DbgHelpSymContainer::
_getStackSymbols(ConstArrayView<StackFrame> frames)
{
  StringBuilder sb;
  _init();
  if (!m_is_good)
    return String();
  int nb_frame = frames.size();
  for (USHORT i = 0; i < nb_frame; i++) {
    DWORD64 addr = (DWORD64)frames[i].address();
    // Allocate a buffer large enough to hold the symbol information on the stack and get
    // a pointer to the buffer.  We also have to set the size of the symbol structure itself
    // and the number of bytes reserved for the name.
    ULONG64 buffer[(sizeof(SYMBOL_INFO) + 1024 + sizeof(ULONG64) - 1) / sizeof(ULONG64)] = { 0 };
    SYMBOL_INFO* info = (SYMBOL_INFO*)buffer;
    info->SizeOfStruct = sizeof(SYMBOL_INFO);
    info->MaxNameLen = 1024;
 
    // Attempt to get information about the symbol and add it to our output parameter.
    DWORD64 displacement = 0;
    if (::SymFromAddr(::GetCurrentProcess(), addr, &displacement, info)) {

      // Attempt to retrieve line number information.
      DWORD line_displacement = 0;
      IMAGEHLP_LINE64 line = {};
      line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
      BOOL has_line = SymGetLineFromAddr64(GetCurrentProcess(), addr, &line_displacement, &line);
      sb.append(std::string_view(info->Name, info->NameLen));
      sb.append("()");
      if (has_line) {
        sb.append(" ");
        sb.append(line.FileName);
        sb.append(":");
        sb.append(String::fromNumber(line.LineNumber));
      }
      sb.append("\n");
    }
  }
  return sb.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  std::shared_ptr<DbgHelpSymContainer> m_sym_container;
  std::shared_ptr<DbgHelpSymContainer> _getStaticContainer()
  {
    if (!m_sym_container.get())
      m_sym_container = std::make_shared<DbgHelpSymContainer>();
    return m_sym_container;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de trace des appels de fonctions utilisant 'DbgHelp'.
 */
class DbgHelpStackTraceService
: public TraceAccessor
, public IStackTraceService
{
 public:

  explicit DbgHelpStackTraceService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {
  }
  explicit DbgHelpStackTraceService(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

   void build() {}

 public:

   StackTrace stackTrace(int first_function) override;
   StackTrace stackTraceFunction(int function_index) override;

 private:
   std::shared_ptr<DbgHelpSymContainer> m_sym_container;
   DbgHelpSymContainer* _getContainer()
   {
     if (!m_sym_container.get())
       m_sym_container = _getStaticContainer();
     return m_sym_container.get();
   }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTrace DbgHelpStackTraceService::
stackTrace(int first_function)
{
  DbgHelpSymContainer* c = _getContainer();
  FixedStackFrameArray frames = c->getStackFrames(first_function);
  String text = c->getStackSymbols(frames.view());
  return StackTrace(frames,text);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StackTrace DbgHelpStackTraceService::
stackTraceFunction(int function_index)
{
  return StackTrace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de trace des appels de fonctions utilisant la libunwind.
 */
class DbgHelpSymbolizerService
: public TraceAccessor
, public ISymbolizerService
{
 public:

  explicit DbgHelpSymbolizerService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {}
  explicit DbgHelpSymbolizerService(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void build() {}

 public:

  String stackTrace(ConstArrayView<StackFrame> frames) override
  {
    return _getContainer()->getStackSymbols(frames);
  }

 private:
  std::shared_ptr<DbgHelpSymContainer> m_sym_container;
  DbgHelpSymContainer* _getContainer()
  {
    if (!m_sym_container.get())
      m_sym_container = _getStaticContainer();
    return m_sym_container.get();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(DbgHelpStackTraceService,
                        ServiceProperty("DbgHelpStackTraceService",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IStackTraceService));

ARCANE_REGISTER_SERVICE(DbgHelpSymbolizerService,
                        ServiceProperty("DbgHelpSymbolizerService",ST_Application),
                        ARCANE_SERVICE_INTERFACE(ISymbolizerService));

ARCANE_DI_REGISTER_PROVIDER(DbgHelpStackTraceService,
                            DependencyInjection::ProviderProperty("DbgHelpStackTraceService"),
                            ARCANE_DI_INTERFACES(IStackTraceService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

ARCANE_DI_REGISTER_PROVIDER(DbgHelpSymbolizerService,
                            DependencyInjection::ProviderProperty("DbgHelpSymbolizerService"),
                            ARCANE_DI_INTERFACES(ISymbolizerService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
