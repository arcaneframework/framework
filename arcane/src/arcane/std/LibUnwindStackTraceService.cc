// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LibUnwindStackTraceService.cc                               (C) 2000-2025 */
/*                                                                           */
/* Function call trace service using 'libunwind'.                            */
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
 * \brief Handler around 'libdw' to retrieve debug information.
 *
 * This allows retrieving the source file name and the line corresponding
 * to a memory address.
 *
 * \note This only works with GCC and not with Clang because the latter
 * does not save debug information in the same way.
 *
 * \warning Calls to 'libdw' functions are probably not re-entrant. They are
 * therefore protected by a mutex.
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
   * \brief Returns the pair (line, column) corresponding to the address \a func_address.
   *
   * If no debug information is found, (line, column) is filled with (0,0)
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
      // WARNING: the returned pointer is no longer valid if m_session is destroyed.
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
    // TODO: call dwfl_end()
    if (!m_session)
      return;

    dwfl_report_begin(m_session);
    dwfl_linux_proc_report(m_session, getpid());
    dwfl_report_end(m_session, nullptr, nullptr);
  }
};

#else

// Empty implementation if libdw is not found.
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
 * \brief Function call trace service using libunwind.
 */
class LibUnwindStackTraceService
: public TraceAccessor
, public IStackTraceService
{
 private:

  //! Information about a memory address.
  struct ProcInfo
  {
   public:

    ProcInfo() = default;
    explicit ProcInfo(const String& aname)
    : m_name(aname)
    {}

   public:

    //! Function name
    const String& name() const { return m_name; }
    //! Name of the library containing the function. May be null.
    const char* libraryFileName() const { return m_library_file_name; }

   public:

    //! Demangled procedure name
    String m_name;
    //! Name of the library (.so or .exe) where the method is located
    const char* m_library_file_name = nullptr;
    //! Library load address.
    //TODO: do not store this for every function.
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

  //! Character string indicating the call stack
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
      // Base address of the file being loaded.
      void* dli_fbase = dl_info.dli_fbase;
      pi.m_library_file_name = dli_fname;
      pi.m_file_loaded_address = (unw_word_t)dli_fbase;
    }
  }

  if (buf)
    pi.m_name = std::string_view(buf);
  else
    pi.m_name = std::string_view(func_name_buf);

  // We must use 'ip-1' because the address returned by libunwind is that
  // of the return instruction.
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
 * \brief Retrieves the function name via dladdr.
 * \todo: make thread-safe
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
    // Error in dladdr.
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

//! Character string indicating the call stack
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
    // It is better to read the call stack from _backtrace() because
    // libunwind returns the return address for each function
    // which causes a shift in the source code line info
    // (it points to the line after the one being executed).
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
 * \brief Returns the current call stack.
 *
 * The first \a function_index functions of the stack are ignored.
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

//! Call stack via the backtrace function.
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

//! Call stack via the backtrace function.
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
 * \brief Generates a list of file names and offsets for a call stack.
 *
 * Generates a character string containing for each address in
 * the call stack \a stack_frames the name of the file containing the symbol
 * and the offset of this address in that file.
 *
 * The output format allows reading by tools such as addr2line
 * or llmv-symbolizer.
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
    // On Linux (CentOS 6 and 7), the address 0x400000 corresponds to that
    // of the executable load, but if we subtract this address from that
    // of the function then the symbolizer does not work (whether it is
    // llvm-symbolize or addr2line)
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

  //! Checks that the specified path is valid
  void _checkValid()
  {
    if (m_is_check_done)
      return;
    // Before calling this method, m_llvm_symbolizer_path must contain
    // the name of the directory where llvm-symbolizer is located.
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
  // NOTE: the code below is similar to
  // LibUnwindStackTraceService::_generateFileAndOffset(). It should
  // merge the two.
  for( Integer i=0, n=frames.size(); i<n; ++i ){
    Dl_info dl_info;
    intptr_t addr = frames[i].address();
    int r2 = dladdr((void*)addr,&dl_info);
    const char* dli_fname = nullptr;
    intptr_t base_address = 0;
    if (r2!=0){
      dli_fname = dl_info.dli_fname;
      // File load base address.
      void* dli_fbase = dl_info.dli_fbase;
      intptr_t true_base = reinterpret_cast<intptr_t>(dli_fbase);
      // On Linux (CentOS 6 and 7), the address 0x400000 corresponds to that
      // of the executable load, but if we subtract this address from that
      // of the function then the symbolizer does not work (whether it is
      // llvm-symbolize or addr2line)
      if (true_base==0x400000)
        true_base = 0;
      base_address = addr - true_base;
    }
    // TODO: write base_address in hex to be able to read it back with addr2line
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
