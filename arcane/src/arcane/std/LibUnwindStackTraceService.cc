// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LibUnwindStackTraceService.cc                               (C) 2000-2025 */
/*                                                                           */
/* Service de trace des appels de fonctions utilisant 'libunwind'.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/StackTrace.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ISymbolizerService.h"

#include "arccore/base/internal/DependencyInjection.h"
#include "arccore/common/internal/Process.h"

#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/Directory.h"

#include "arcane/core/ServiceFactory.h"
#include "arcane/core/AbstractService.h"

#include "arcane_packages.h"

#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <cxxabi.h>

#include <map>
#include <mutex>

#include <execinfo.h>
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>
//#include <cstdlib>
#include <dlfcn.h>

#if defined(ARCANE_HAS_PACKAGE_DW)
#include <elfutils/libdwfl.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

struct DebugSourceInfo
{
  int m_line = 0;
  int m_column = 0;
  const char* m_source_file = nullptr;
};

#if defined(ARCANE_HAS_PACKAGE_DW)
/*!
 * \brief Handler autour de 'libdw' pour récupérer les informations de debug.
 *
 * Cela permet de récupérer le nom du fichier source et la ligne correspondant
 * à une adresse mémoire.
 *
 * \note Cela ne fonctionne qu'avec GCC et pas avec Clang car ce dernier ne
 * sauve pas de la même manière les informations de debug.
 *
 * \warning Les appels aux fonctions de 'libdw' ne sont probablement pas
 * ré-entrant. Ils sont donc protégés par un mutex.
 */
class DWHandler
{
 public:

  DWHandler()
  {
    m_callback.find_elf = &dwfl_linux_proc_find_elf;
    m_callback.find_debuginfo = &dwfl_standard_find_debuginfo;
    m_callback.section_address = nullptr;
    m_callback.debuginfo_path = nullptr;
  }

  ~DWHandler()
  {
    if (m_session)
      dwfl_end(m_session);
  }

  /*!
   * \brief Retourne le couple (ligne,colonne) correspondant à l'adresse \a func_address.
   *
   * Si aucune information de debug n'est trouvée, remplit (line,column) par (0,0)
   */
  DebugSourceInfo getInfo(unw_word_t func_address)
  {
    std::scoped_lock<std::mutex> lock(m_mutex);
    DebugSourceInfo source_info;

    _init();

    auto dw_address = reinterpret_cast<Dwarf_Addr>(func_address);

    Dwfl_Module* module = dwfl_addrmodule(m_session, dw_address);
    if (!module)
      return source_info;
    Dwarf_Addr bias = 0;
    Dwarf_Die* dw_die = dwfl_module_addrdie(module, dw_address, &bias);
    if (dw_die)
      return source_info;

    Dwarf_Line* dw_source_info = dwarf_getsrc_die(dw_die, dw_address - bias);

    if (dw_source_info) {
      int line = 0;
      int column = 0;
      // ATTENTION : le pointeur retourné n'est plus valide si m_session est détruit.
      const char* source_file = dwarf_linesrc(dw_source_info, nullptr, nullptr);
      dwarf_lineno(dw_source_info, &line);
      dwarf_linecol(dw_source_info, &column);
      source_info = { line, column, source_file };
    }

    return source_info;
  }

 public:

  bool m_is_init = false;
  Dwfl_Callbacks m_callback;
  Dwfl* m_session = nullptr;
  std::mutex m_mutex;

 private:

  void _init()
  {
    if (m_is_init)
      return;
    m_is_init = true;

    m_session = dwfl_begin(&m_callback);
    // TODO: faire dwfl_end()
    if (!m_session)
      return;

    dwfl_report_begin(m_session);
    dwfl_linux_proc_report(m_session, getpid());
    dwfl_report_end(m_session, nullptr, nullptr);
  }
};

#else

// Implémentation vide si libdw n'est pas trouvé.
class DWHandler
{
 public:

  DebugSourceInfo getInfo(unw_word_t func_address)
  {
    return {};
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de trace des appels de fonctions utilisant la libunwind.
 */
class LibUnwindStackTraceService
: public TraceAccessor
, public IStackTraceService
{
 private:

  //! Information sur une adresse mémoire.
  struct ProcInfo
  {
   public:

    ProcInfo() = default;
    explicit ProcInfo(const String& aname)
    : m_name(aname)
    {}

   public:

    //! Nom de la fonction
    const String& name() const { return m_name; }
    //! Nom de la bibliothèque contenant la fonction. Peut-être nul.
    const char* libraryFileName() const { return m_library_file_name; }

   public:

    //! Nom (démanglé) de la procédure
    String m_name;
    //! Nom de la bibliothèque (.so ou .exe) dans laquelle se trouve la méthode
    const char* m_library_file_name = nullptr;
    //! Adresse de chargement de la bibliothèque.
    //TODO: ne pas stocker cela pour chaque fonction.
    unw_word_t m_file_loaded_address = 0;
    unw_word_t m_base_ip = 0;
    DebugSourceInfo m_source_info;
  };

 public:

  explicit LibUnwindStackTraceService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {
  }
  explicit LibUnwindStackTraceService(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void build() override
  {
    if (!platform::getEnvironmentVariable("ARCANE_GDB_STACK").null())
      m_want_gdb_info = true;
    if (!platform::getEnvironmentVariable("ARCANE_USE_BACKTRACE").null())
      m_use_backtrace = true;
  }

 public:

  //! Chaîne de caractère indiquant la pile d'appel
  StackTrace stackTrace(int first_function) override;
  StackTrace stackTraceFunction(int function_index) override;

 private:

  using ProcInfoMap = std::map<unw_word_t,ProcInfo>;
  ProcInfoMap m_proc_name_map;
  std::mutex m_proc_name_map_mutex;

  bool m_want_gdb_info = false;
  bool m_use_backtrace = false;
  DWHandler m_dw_handler;
  ProcInfo _getFuncInfo(unw_word_t ip,unw_cursor_t* cursor);
  ProcInfo _getFuncInfo(const void* ip);
  String _getGDBStack();
  StackTrace _backtraceStackTrace(const FixedStackFrameArray& stack_frames);
  String _generateFileAndOffset(const FixedStackFrameArray& stack_frames);
  FixedStackFrameArray _backtraceStackFrame(int first_function);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LibUnwindStackTraceService::ProcInfo LibUnwindStackTraceService::
_getFuncInfo(unw_word_t ip,unw_cursor_t* cursor)
{
  {
    std::lock_guard<std::mutex> lk(m_proc_name_map_mutex);
    auto v = m_proc_name_map.find(ip);
    if (v!=m_proc_name_map.end())
      return v->second;
  }

  unw_word_t offset;
  char func_name_buf[10000];
  char demangled_func_name_buf[10000];
  unw_get_proc_name(cursor,func_name_buf,10000,&offset);
  int dstatus = 0;
  size_t len = 10000;
  char* buf = abi::__cxa_demangle (func_name_buf,demangled_func_name_buf,&len,&dstatus);
  ProcInfo pi;
  pi.m_base_ip = offset;
  {
    Dl_info dl_info;
    void* addr = (void*)ip;
    int r2 = dladdr(addr,&dl_info);
    if (r2!=0){
      const char* dli_fname = dl_info.dli_fname;
      // Adresse de base de chargement du fichier.
      void* dli_fbase = dl_info.dli_fbase;
      pi.m_library_file_name = dli_fname;
      pi.m_file_loaded_address = (unw_word_t)dli_fbase;
    }
  }

  if (buf)
    pi.m_name = std::string_view(buf);
  else
    pi.m_name = std::string_view(func_name_buf);

  // Il faut faire 'ip-1' car l'adresse retournée par libunwind est celle
  // de l'instruction de retour.
  pi.m_source_info = m_dw_handler.getInfo(ip - 1);

  {
    std::lock_guard<std::mutex> lk(m_proc_name_map_mutex);
    m_proc_name_map.insert(ProcInfoMap::value_type(ip, pi));
  }

  return pi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Récupère le nom de la fonction via dladdr.
 * \todo: rendre thread-safe
 */
LibUnwindStackTraceService::ProcInfo LibUnwindStackTraceService::
_getFuncInfo(const void* addr)
{
  {
    std::lock_guard<std::mutex> lk(m_proc_name_map_mutex);
    auto v = m_proc_name_map.find((unw_word_t)addr);
    if (v!=m_proc_name_map.end()){
      return v->second;
    }
  }
  const size_t buf_size = 10000;
  char demangled_func_name_buf[buf_size];
  Dl_info dl_info;
  int r = dladdr(addr, &dl_info);
  if (r==0){
    // Erreur dans dladdr.
    std::cout << "ERROR in dladdr\n";
    return ProcInfo("Unknown");
  }
  const char* dli_sname = dl_info.dli_sname;
  if (!dli_sname)
    dli_sname = "";
  int dstatus = 0;
  size_t len = buf_size;
  char* buf = abi::__cxa_demangle (dli_sname,demangled_func_name_buf,&len,&dstatus);
  //char* buf = (char*)dli_sname;
  ProcInfo pi;
  if (buf)
    pi.m_name = std::string_view(buf);
  else
    pi.m_name = std::string_view(dli_sname);
  {
    std::lock_guard<std::mutex> lk(m_proc_name_map_mutex);
    m_proc_name_map.insert(ProcInfoMap::value_type((unw_word_t)addr,pi));
  }
  return pi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String LibUnwindStackTraceService::
_getGDBStack()
{
  void *array [256];
  char **names;
  int i, size;

  fprintf (stderr, "\nNative stacktrace:\n\n");

  size = backtrace (array, 256);
  names = backtrace_symbols (array, size);
  for (i =0; i < size; ++i) {
    fprintf (stderr, "\t%s\n", names [i]);
  }

  fflush (stderr);
  return platform::getGDBStack();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Chaîne de caractère indiquant la pile d'appel
StackTrace LibUnwindStackTraceService::
stackTrace(int first_function)
{
  unw_cursor_t cursor;
  unw_context_t uc;

  String last_str;
  if (m_want_gdb_info)
    last_str = _getGDBStack();
  FixedStackFrameArray backtrace_stack_frames = _backtraceStackFrame(first_function);
  if (m_use_backtrace)
    return _backtraceStackTrace(backtrace_stack_frames);

  const size_t hexa_buf_size = 100;
  char hexa[hexa_buf_size+1];

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  int current_func = 0;
  StringBuilder message;

  FixedStackFrameArray stack_frames;
  while (unw_step(&cursor) > 0) {
    unw_word_t ip;
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    if (current_func>=first_function){
      ProcInfo pi = _getFuncInfo(ip,&cursor);
      String func_name = pi.m_name;
      message += " ";
      snprintf(hexa,hexa_buf_size,"%14llx",(long long)ip);
      message += hexa;
      message += "  ";
      message += func_name;
      if (pi.m_source_info.m_source_file) {
        message += " \"";
        message += pi.m_source_info.m_source_file;
        if (pi.m_source_info.m_line > 0) {
          message += ":";
          message += pi.m_source_info.m_line;
        }
        message += "\"";
      }
      message += "\n";

      stack_frames.addFrame(StackFrame((intptr_t)ip));
    }
    ++current_func;
  }
  ISymbolizerService* ss = platform::getSymbolizerService();

  if (ss){
    // Il faut mieux lire la pile d'appel de _backtrace() car
    // la libunwind retourne l'adresse de retour pour chaque fonction
    // ce qui provoque un décalage dans les infos des lignes du code
    // source (on pointe vers la ligne après celle qu'on est en train d'exécuter).
    last_str = ss->stackTrace(backtrace_stack_frames.view());
  }
  else{
    message += "\nFileAndOffsetStack:{{\n";
    message += _generateFileAndOffset(backtrace_stack_frames);
    message += "}}\n";
  }
  message += last_str;
  return StackTrace(stack_frames,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la pile à d'appel actuelle.
 *
 * Les \a function_index premières fonctions de la pile sont ignorées.
 */
StackTrace LibUnwindStackTraceService::
stackTraceFunction(int function_index)
{
  unw_cursor_t cursor;
  unw_context_t uc;
  unw_word_t ip;
  //unw_word_t offset;

  String last_str;
  if (m_want_gdb_info)
    last_str = _getGDBStack();

  unw_getcontext(&uc);
  unw_init_local(&cursor, &uc);
  int current_func = 0;
  StringBuilder message;

  while (unw_step(&cursor) > 0) {
    unw_get_reg(&cursor, UNW_REG_IP, &ip);
    if (current_func==function_index){
      ProcInfo pi = _getFuncInfo(ip,&cursor);
      String func_name = pi.m_name;
      message += func_name;
      break;
    }
    ++current_func;
  }
  return StackTrace(message.toString()+last_str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Pile d'appel via la fonction backtrace.
FixedStackFrameArray LibUnwindStackTraceService::
_backtraceStackFrame(int first_function)
{
	void *ips [256];
	Integer size = backtrace (ips, 256);

  FixedStackFrameArray stack_frames;
  for( Integer i=first_function; i<size; ++i ){
    stack_frames.addFrame(StackFrame((intptr_t)ips[i]));
  }
  return stack_frames;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Pile d'appel via la fonction backtrace.
StackTrace LibUnwindStackTraceService::
_backtraceStackTrace(const FixedStackFrameArray& stack_frames)
{
  const size_t buf_size = 100;
  char hexa[buf_size+1];

  StringBuilder message;
  ConstArrayView<StackFrame> frames_view = stack_frames.view();
  for( StackFrame f : frames_view ){
    intptr_t ip = f.address();
    ProcInfo pinfo = _getFuncInfo((void*)ip);
    String func_name = pinfo.name();
    message += "  ";
    snprintf(hexa,buf_size,"%10llx",(long long)ip);
    message += hexa;
    message += "  ";
    message += func_name;
    message += "\n";
  }
  return StackTrace(stack_frames,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Génère liste des noms de fichier et offset d'une pile d'appel.
 *
 * Génère une chaîne de caractère contenant pour chaque adresse de
 * la pile d'appel \a stack_frames le nom du fichier contenant le symbol
 * et l'offset de cette adresse dans ce fichier.
 *
 * Le format de sortie permet la lecture par des outils tels que addr2line
 * ou llmv-symbolizer.
 */
String LibUnwindStackTraceService::
_generateFileAndOffset(const FixedStackFrameArray& stack_frames)
{
  const size_t buf_size = 100;
  char hexa[buf_size+1];

  StringBuilder message;
  ConstArrayView<StackFrame> frames_view = stack_frames.view();
  for( StackFrame f : frames_view ){
    intptr_t ip = f.address();
    ProcInfo pinfo = _getFuncInfo(reinterpret_cast<const void*>(ip));
    message += (pinfo.libraryFileName() ? pinfo.libraryFileName() : "()");
    message += "  ";
    auto file_base_address = pinfo.m_file_loaded_address;
    // Sous Linux (CentOS 6 et 7), l'adresse 0x400000 correspond à celle
    // de chargement de  l'exécutable mais si on retranche cette adresse de celle
    // de la fonction alors le symboliser ne fonctionne pas (que ce soit
    // llvm-symbolize ou addr2line)
    if (file_base_address==0x400000)
      file_base_address = 0;
    intptr_t offset_ip = (ip - file_base_address);
    snprintf(hexa,buf_size,"%llx",(long long)offset_ip);
    message += "0x";
    message += hexa;
    message += "\n";
  }
  return message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LLVMSymbolizerService
: public TraceAccessor
, public ISymbolizerService
{
 public:

  explicit LLVMSymbolizerService(const ServiceBuildInfo& sbi)
  : TraceAccessor(sbi.application()->traceMng())
  {
    _init();
  }
  explicit LLVMSymbolizerService(ITraceMng* tm)
  : TraceAccessor(tm)
  {
    _init();
  }

 public:

  void build() {}

 public:

  String stackTrace(ConstArrayView<StackFrame> frames) override;

 private:

  String m_llvm_symbolizer_path;
  bool m_is_check_done = false;
  bool m_is_valid = false;

 private:

  //! Vérifie que le chemin spécifié est valid
  void _checkValid()
  {
    if (m_is_check_done)
      return;
    // Avant appel à cette méthode, m_llvm_symbolizer_path doit contenir
    // le nom du répertoire dans lequel se trouve llvm-symbolizer.
    Directory dir(m_llvm_symbolizer_path);
    String fullpath = dir.file("llvm-symbolizer");
    Int64 length = platform::getFileLength(fullpath);
    m_llvm_symbolizer_path = fullpath;
    if (length>0)
      m_is_valid = true;
    m_is_check_done = true;
  }
  void _init()
  {
    m_llvm_symbolizer_path = platform::getEnvironmentVariable("ARCANE_LLVMSYMBOLIZER_PATH");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String LLVMSymbolizerService::
stackTrace(ConstArrayView<StackFrame> frames)
{
  _checkValid();
  if (!m_is_valid)
    return String();
  std::stringstream ostr;
  // NOTE: le code ci dessous est similaire à
  // LibUnwindStackTraceService::_generateFileAndOffset(). Il faudrait
  // fusionner les deux.
  for( Integer i=0, n=frames.size(); i<n; ++i ){
    Dl_info dl_info;
    intptr_t addr = frames[i].address();
    int r2 = dladdr((void*)addr,&dl_info);
    const char* dli_fname = nullptr;
    intptr_t base_address = 0;
    if (r2!=0){
      dli_fname = dl_info.dli_fname;
      // Adresse de base de chargement du fichier.
      void* dli_fbase = dl_info.dli_fbase;
      intptr_t true_base = reinterpret_cast<intptr_t>(dli_fbase);
      // Sous Linux (CentOS 6 et 7), l'adresse 0x400000 correspond à celle
      // de chargement de l'exécutable, mais si on retranche cette adresse de celle
      // de la fonction alors le symboliser ne fonctionne pas (que ce soit
      // llvm-symbolize ou addr2line)
      if (true_base==0x400000)
        true_base = 0;
      base_address = addr - true_base;
    }
    // TODO: écrire base_address en hexa pour pouvoir le relire avec addr2line
    ostr << (dli_fname ? dli_fname : "??") << " " << base_address << '\n';
  }

  std::string input_str(ostr.str());
  String output_str;
  ProcessExecArgs args;
  args.setCommand(m_llvm_symbolizer_path);
  Integer input_size = arcaneCheckArraySize(input_str.length());
  ByteConstArrayView input_bytes(input_size, reinterpret_cast<const Byte*>(input_str.c_str()));
  args.setInputBytes(input_bytes);
  args.addArguments("--demangle");
  args.addArguments("--pretty-print");
  //args.addArguments("--print-source-context-lines=5");
  Process::execute(args);
  return String(args.outputBytes());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LibUnwindStackTraceService,
                        ServiceProperty("LibUnwind",ST_Application),
                        ARCANE_SERVICE_INTERFACE(IStackTraceService));

ARCANE_REGISTER_SERVICE(LLVMSymbolizerService,
                        ServiceProperty("LLVMSymbolizer",ST_Application),
                        ARCANE_SERVICE_INTERFACE(ISymbolizerService));

ARCANE_DI_REGISTER_PROVIDER(LibUnwindStackTraceService,
                            DependencyInjection::ProviderProperty("LibUnwind"),
                            ARCANE_DI_INTERFACES(IStackTraceService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

ARCANE_DI_REGISTER_PROVIDER(LLVMSymbolizerService,
                            DependencyInjection::ProviderProperty("LLVMSymbolizer"),
                            ARCANE_DI_INTERFACES(ISymbolizerService),
                            ARCANE_DI_CONSTRUCTOR(ITraceMng*));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
