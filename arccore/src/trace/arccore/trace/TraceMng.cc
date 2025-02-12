// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceMng.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des traces.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/ITraceMng.h"
#include "arccore/trace/TraceClassConfig.h"
#include "arccore/trace/StandaloneTraceMessage.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/IFunctor.h"
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/Span.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arccore/concurrency/Mutex.h"

#include <sstream>
#include <fstream>
#include <limits>
#include <set>
#include <map>
#include <vector>
#include <memory>

#include <time.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// TODO réimplémenter cette classe
class TraceTimer
{
 public:
  double getTime() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fichier ou flux de traces.
 */
class FileTraceStream
: public ITraceStream
{
 public:
  explicit FileTraceStream(const String& filename)
  : m_nb_ref(0), m_stream(nullptr), m_need_destroy(true)
  {
    m_stream = new std::ofstream(filename.localstr());
  }
  FileTraceStream(std::ostream* stream,bool need_destroy)
  : m_nb_ref(0), m_stream(stream), m_need_destroy(need_destroy)
  {
  }
  ~FileTraceStream() override
  {
    if (m_need_destroy)
      delete m_stream;
  }
 public:
  void addReference() override { ++m_nb_ref; }
  void removeReference() override
  {
    // Décrémente et retourne la valeur d'avant.
    // Si elle vaut 1, cela signifie qu'on n'a plus de références
    // sur l'objet et qu'il faut le détruire.
    Int32 v = std::atomic_fetch_add(&m_nb_ref,-1);
    if (v==1)
      delete this;
  }
  std::ostream* stream() override { return m_stream; }
 private:
  std::atomic<Int32> m_nb_ref;
  std::ostream* m_stream;
  bool m_need_destroy;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceStream* ITraceStream::
createFileStream(const String& filename)
{
  return new FileTraceStream(filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceStream* ITraceStream::
createStream(std::ostream* stream,bool need_destroy)
{
  return new FileTraceStream(stream,need_destroy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère une liste de flux par thread.
 *
 * Il ne doit y avoir qu'une seule instance de cette classe par thread.
 *
 * Cette classe permet de garantir que les affichages listing par thread
 * se font correctement.
 */
class TraceMngStreamList
{
 public:
  // A mettre en correspondance avec Trace::Trace::eMessageType
  static const Integer NB_STREAM = 9;
  TraceMngStreamList()
  {
    m_str_list[Trace::Normal] = &m_str_std;
    m_str_list[Trace::Info] = &m_str_info;
    m_str_list[Trace::Warning] = &m_str_warning;
    m_str_list[Trace::Error] = &m_str_error;
    m_str_list[Trace::Log] = &m_str_log;
    m_str_list[Trace::Fatal] = &m_str_fatal;
    m_str_list[Trace::ParallelFatal] = &m_str_parallel_fatal;
    m_str_list[Trace::Debug] = &m_str_debug;
    m_str_list[Trace::Null] = &m_str_null;

    for( Integer i=0; i<NB_STREAM; ++i ){
      m_str_count[i] = 0;
      m_str_list[i]->precision(std::numeric_limits<Real>::digits10);
    }
  }
 public:
  std::ostringstream m_str_std;
  std::ostringstream m_str_info;
  std::ostringstream m_str_warning;
  std::ostringstream m_str_error;
  std::ostringstream m_str_log;
  std::ostringstream m_str_fatal;
  std::ostringstream m_str_parallel_fatal;
  std::ostringstream m_str_debug;
  std::ostringstream m_str_null;
  std::ostringstream* m_str_list[NB_STREAM];
  Integer m_str_count[NB_STREAM];
  std::ostringstream m_tmp_buf;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conteneur pour gérer les instances de TraceMngStreamList.
 */
class TraceMngStreamListStorage
{
 public:
  TraceMngStreamListStorage() ARCCORE_NOEXCEPT : m_str_list(nullptr){}
  ~TraceMngStreamListStorage()
  {
    delete m_str_list;
  }
  TraceMngStreamList* item()
  {
    if (!m_str_list){
      m_str_list = new TraceMngStreamList();
    }
    return m_str_list;
  }
 private:
  TraceMngStreamList* m_str_list;
};

namespace
{
thread_local TraceMngStreamListStorage global_stream_list_storage;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation du gestionnaire de traces.
 */
class TraceMng
: public ReferenceCounterImpl
, public ITraceMng
{
 public:

  TraceMng();

 protected:

  ~TraceMng() override;

 public:

  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:
	
  TraceMessage operator()()
  {
    return TraceMessage(_getStream(Trace::Normal),this,Trace::Normal);
  }
  TraceMessage info() override
  {
    return (_isCurrentClassActivated()) ? _info() : _devNull();
  }
  TraceMessage info(char /*category*/) override
  {
    return (_isCurrentClassActivated()) ? _info() : _devNull();
  }
  TraceMessage info(Int32 verbose_level) override
  {
    return (_isCurrentClassActivated()) ? _info(verbose_level) : _devNull();
  }
  TraceMessage pinfo() override
  {
    return (_isCurrentClassParallelActivated()) ? _info(0) : _devNull();
  }
  TraceMessage pinfo(char /*category*/) override
  {
    return (_isCurrentClassParallelActivated()) ? _info(0) : _devNull();
  }
  TraceMessage info(bool is_ok) override
  {
    return (is_ok) ? _info() : _devNull();
  }
  TraceMessage warning() override
  {
    return TraceMessage(_getStream(Trace::Warning),this,Trace::Warning);
  }
  TraceMessage pwarning() override
  {
    return m_is_master ? warning() : _devNull();
  }
  TraceMessage error() override
  {
    return TraceMessage(_getStream(Trace::Error),this,Trace::Error);
  }
  TraceMessage perror() override
  {
    return m_is_master ? error() : _devNull();
  }
  TraceMessage log() override
  {
    return (_isCurrentClassActivated()) ? _log(false) : _devNull();
  }
  TraceMessage plog() override
  {
    return (_isCurrentClassParallelActivated()) ? _log(false) : _devNull();
  }
  TraceMessage logdate() override
  {
    return (_isCurrentClassActivated()) ? _log(true) : _devNull();
  }
  TraceMessage fatal() override
  {
    return TraceMessage(_getStream(Trace::Fatal),this,Trace::Fatal);
  }
  TraceMessage pfatal() override
  {
    return TraceMessage(_getStream(Trace::ParallelFatal),this,Trace::ParallelFatal);
  }
  TraceMessage devNull() override
  {
    return _devNull();
  }
  TraceMessageDbg debug(Trace::eDebugLevel dbg_lvl) override
  {
    return (dbg_lvl<=_configDbgLevel()) ? _dbg() : _dbgDevNull();
  }
  void endTrace(const TraceMessage* msg) override;
  void beginTrace(const TraceMessage* msg) override;
  void putTrace(const String& message,int type) override;
	
  void addListener(ITraceMessageListener* v) override;
  void removeListener(ITraceMessageListener* v) override;

  bool setInfoActivated(bool v) override
  {
    bool old = m_is_info_activated;
    m_is_info_activated = v;
    return old;
  }
  bool isInfoActivated() const override { return m_is_info_activated; }

  void finishInitialize() override;

  void pushTraceClass(const String& name) override;
  void popTraceClass() override;

  void flush() override;

  void setRedirectStream(std::ostream* ro) override
  {
    m_listing_stream = new FileTraceStream(ro,false);
  }
  void setRedirectStream(ITraceStream* stream) override
  {
    m_listing_stream = stream;
  }

  Trace::eDebugLevel configDbgLevel() const override { return _configDbgLevel(); }

  void setErrorFileName(const String& file_name) override;
  void setLogFileName(const String& file_name) override;

  void setClassConfig(const String& name,const TraceClassConfig& config) override;
  TraceClassConfig classConfig(const String& name) const override;
  void removeAllClassConfig() override;

  void setMaster(bool is_master) override;
  bool isMaster() const override { return m_is_master; }

  void setVerbosityLevel(Int32 level) override;
  Int32 verbosityLevel() const override { return m_verbosity_level; }

  void setStandardOutputVerbosityLevel(Int32 level) override;
  Int32 standardOutputVerbosityLevel() const override { return m_stdout_verbosity_level; }

  void resetThreadStatus() override;

  void writeDirect(const TraceMessage* msg,const String& str) override;

  void setTraceId(const String& id) override { m_trace_id = id; }
  const String& traceId() const override { return m_trace_id; }

  void visitClassConfigs(IFunctorWithArgumentT<std::pair<String,TraceClassConfig>>* functor) override;

 protected:
	
  TraceMessage _log(bool print_date)
  {
    if (print_date){
      TraceMessage msg(_getStream(Trace::Log),this,Trace::Log);
      _putDate(msg.file());
      return msg;
    }
    return TraceMessage(_getStream(Trace::Log),this,Trace::Log);
  }
  TraceMessage _info()
  {
    return TraceMessage(_getStream(Trace::Info),this,Trace::Info);
  }
  TraceMessage _info(Int32 verbose_level)
  {
    return TraceMessage(_getStream(Trace::Info),this,Trace::Info,verbose_level);
  }
  TraceMessage _devNull()
  {
    return TraceMessage(_getStream(Trace::Null),this,Trace::Null);
  }
#ifdef ARCCORE_DEBUG
  TraceMessageDbg _dbg()
  {
    return TraceMessage(_getStream(Trace::Debug),this,Trace::Debug);
  }
  TraceMessageDbg _dbgDevNull()
  {
    return TraceMessage(_getStream(Trace::Null),this,Trace::Null);
  }
#else
  TraceMessageDbg _dbg() { return TraceMessageDbg(); }
  TraceMessageDbg _dbgDevNull() { return TraceMessageDbg(); }
#endif

  bool _isCurrentClassActivated() const
  {
    bool v = false;
    {
      Mutex::ScopedLock sl(m_trace_mutex);
      v = m_current_msg_class.m_info->isActivated();
    }
    return v;
  }

  bool _isCurrentClassParallelActivated() const
  {
    bool v = false;
    {
      Mutex::ScopedLock sl(m_trace_mutex);
      v = m_current_msg_class.m_info->isParallelActivated();
    }
    return v;
  }

  Trace::eDebugLevel _configDbgLevel() const
  {
    Trace::eDebugLevel dbg_level;
    {
      Mutex::ScopedLock sl(m_trace_mutex);
      dbg_level = m_current_msg_class.m_info->debugLevel();
    }
    return dbg_level;
  }


 private:

  /*!
   * \brief Information sur une classe de messages.
   */
  class TraceClass
  {
   public:
    TraceClass(const String& name,const TraceClassConfig* mci)
    : m_name(name), m_info(mci) {}
   public:
    String m_name;
    const TraceClassConfig* m_info; //!< Configuration de sortie des informations
  };

 private:

  typedef std::set<ITraceMessageListener*> ListenerList;
  typedef std::vector<TraceClass> TraceClassStack;

  bool m_is_master = true;
  bool m_want_trace_function = false;
  bool m_want_trace_timer = false;
  Int32 m_verbosity_level = TraceMessage::DEFAULT_LEVEL;
  Int32 m_stdout_verbosity_level = TraceMessage::DEFAULT_LEVEL;
  std::atomic<Int32> m_current_class_verbosity_level = TraceMessage::DEFAULT_LEVEL;
  std::atomic<Int32> m_current_class_flags = Trace::PF_Default;
  ListenerList* m_listeners = nullptr;
  bool m_is_info_activated = true;
  std::map<String,TraceClassConfig*> m_trace_class_config_map;
  TraceClassStack m_trace_class_stack;
  TraceClassConfig m_default_trace_class_config;
  TraceClass m_default_trace_class;
  TraceClass m_current_msg_class;
  ReferenceCounter<ITraceStream> m_listing_stream;
  std::atomic<Int64> m_nb_flush = 0;
  String m_error_file_name;
  String m_log_file_name;
  String m_trace_id;
  ReferenceCounter<ITraceStream> m_error_file;
  ReferenceCounter<ITraceStream> m_log_file;
  Mutex* m_trace_mutex = nullptr;
  bool m_is_error_disabled = false;
  bool m_is_log_disabled = false;
  bool m_has_color = false;
  TraceTimer m_trace_timer;

 private:

  void _writeTimeString(std::ostream& out);
  TraceMngStreamList* _getStreamList() const
  {
    return global_stream_list_storage.item();
  }
  std::ostream* _getStream(Trace::eMessageType id)
  {
    int iid = static_cast<int>(id);
    std::ostringstream* ostr = _getStreamList()->m_str_list[iid];
    ostr->str(std::string());
    return ostr;
  }
  bool _sendToProxy2(const TraceMessage* msg,Span<const Byte> str);

  //NOTE: cette méthode doit être appelée avec le verrou \a m_trace_mutex positionné.
  const TraceClassConfig* _msgClassConfig(const String& s) const
  {
    auto ci = m_trace_class_config_map.find(s);
    if (ci!=m_trace_class_config_map.end()){
      return ci->second;
    }
    return &m_default_trace_class_config;
  }
  String _currentTraceClassName() const
  {
    Mutex::ScopedLock sl(m_trace_mutex);
    return m_current_msg_class.m_name;
  }
  void _checkFlush();
  void _putStream(std::ostream& ostr,Span<const Byte> buffer);
  void _putTraceMessage(std::ostream& ostr,Trace::eMessageType id,Span<const Byte>);
  void _putDate(std::ostream& ostr);
  std::ostream* _errorStream();
  std::ostream* _logStream();
  void _write(std::ostream& output,Span<const Byte> input,bool do_flush=false);
  void _writeColor(std::ostream& output,Span<const Byte> input,int color,bool do_flush);
  void _writeListing(Span<const Byte> input,int level,int color,bool do_flush);
  void _write(std::ostream* output,Span<const Byte> input,bool do_flush=false);
  void _writeStackTrace(std::ostream* output,const String& stack_trace);
  void _endTrace(const TraceMessage* msg);
  void _putFunctionName(std::ostream& out);
  void _writeDirect(const TraceMessage* msg,Span<const Byte> buf_array,
                    Span<const Byte> orig_message);
  void _putTraceId(std::ostream& out);
  void _updateCurrentClassConfig();
  void _flushStream(ITraceStream* stream);
  void _writeSpan(std::ostream& o,Span<const Byte> text);
  FileTraceStream* _createFileStream(StringView file_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_TRACE_EXPORT
ITraceMng* arccoreCreateDefaultTraceMng()
{
  return new TraceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceMng::
TraceMng()
: m_default_trace_class("Internal", &m_default_trace_class_config)
, m_current_msg_class(m_default_trace_class)
, m_error_file_name("errors")
, m_trace_mutex(new Mutex())
{
  m_has_color = Platform::getConsoleHasColor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceMng::
~TraceMng()
{
  for( const auto& i : m_trace_class_config_map )
    delete i.second;
  delete m_listeners;
  delete m_trace_mutex;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TraceMng::
_sendToProxy2(const TraceMessage* msg,Span<const Byte> buf)
{
  if (m_listeners){
    // TODO: changer le prototype pour utiliser Span
    const char* buf_data = reinterpret_cast<const char*>(buf.data());
    ConstArrayView<char> cbuf(arccoreCheckArraySize(buf.size()),buf_data);
    TraceMessageListenerArgs args(msg,cbuf);
    for( auto itml : (*m_listeners) ){
      if (itml->visitMessage(args))
        return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream* TraceMng::
_errorStream()
{
  Mutex::ScopedLock sl(m_trace_mutex);
  if (m_is_error_disabled)
    return nullptr;
  if (m_error_file.isNull())
    m_error_file = new FileTraceStream(m_error_file_name);
  return m_error_file->stream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream* TraceMng::
_logStream()
{
  Mutex::ScopedLock sl(m_trace_mutex);
  if (m_is_log_disabled)
    return nullptr;
  if (m_log_file.isNull())
    m_log_file = new FileTraceStream(m_log_file_name);
  return m_log_file->stream();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
flush()
{
  std::cout.flush();
  _flushStream(m_listing_stream.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FileTraceStream* TraceMng::
_createFileStream(Arccore::StringView file_name)
{
  auto x = new FileTraceStream(file_name);
  std::ostream* ostr = x->stream();
  if (!ostr || ostr->bad()) {
    // Ne pas utiliser 'warning()' ou 'error()' car cette méthode peut être
    // appelée lors du positionnement des logs ou des erreurs.
    info() << "WARNING: Can not open file '" << file_name << "' for writing";
  }
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setErrorFileName(const String& file_name)
{
  if (m_error_file_name==file_name)
    return;
  m_error_file_name = file_name;
  m_error_file = nullptr;
  m_is_error_disabled = m_error_file_name.null();
  if (!m_is_error_disabled)
    m_error_file = _createFileStream(file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setLogFileName(const String& file_name)
{
  if (m_log_file_name==file_name)
    return;
  m_log_file_name = file_name;
  m_is_log_disabled = m_log_file_name.null();
  m_log_file = nullptr;
  if (!m_is_log_disabled)
    m_log_file = _createFileStream(file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putDate(std::ostream& ostr)
{
  static const size_t max_len = 80;
  char str[max_len];
  time_t now_time;
  const struct tm* now_tm;
  ::time(&now_time);
  now_tm = ::localtime(&now_time);

  strftime(str,max_len,"[%m/%d/%Y %X] ",now_tm);
  ostr << str;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_flushStream(ITraceStream* stream)
{
  if (!stream)
    return;
  auto f = stream->stream();
  if (f)
    f->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_checkFlush()
{
  std::cout.flush();
  ++m_nb_flush;
  if ( (m_nb_flush % 50) == 0 ){
    m_nb_flush = 0;
    _flushStream(m_listing_stream.get());
    std::cout.flush();
  }
  std::cout.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeSpan(std::ostream& o,Span<const Byte> text)
{
  o.write(reinterpret_cast<const char*>(text.data()),text.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putFunctionName(std::ostream& out)
{
  if (!m_want_trace_function)
    return;
  IStackTraceService* sts = Platform::getStackTraceService();
  if (!sts)
    return;
  String sts_str = sts->stackTraceFunction(4).toString();

  std::string s = sts_str.localstr();
  std::string::size_type spos = s.find('(');
  if (spos!=std::string::npos)
    s.erase(s.begin()+spos,s.end());
  out << s << "() --> ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putTraceId(std::ostream& out)
{
  if (!m_trace_id.null() && !m_trace_id.empty()){
    out << " (" << m_trace_id << ")";
    return;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeTimeString(std::ostream& out)
{
  char trace_timer_buffer[256];
  unsigned long t = static_cast<unsigned long>(m_trace_timer.getTime());
  const unsigned long hour = t/3600; t -= hour*3600;
  const unsigned long min  = t/60;   t -= min*60;
  // TODO utiliser une méthode portable.
#ifdef ARCCORE_OS_WIN32
  _snprintf(trace_timer_buffer,256,"%04lu:%02lu:%02lu",hour,min,t);
#else
  snprintf(trace_timer_buffer,256,"%04lu:%02lu:%02lu",hour,min,t);
#endif
  out << trace_timer_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putTraceMessage(std::ostream& out,Trace::eMessageType id,Span<const Byte> msg_str)
{
  if (m_want_trace_timer || (m_current_class_flags & Trace::PF_ElapsedTime)){
    _writeTimeString(out);
    out << " ";
  }
  switch(id){
   case Trace::Info:
     if (!(m_current_class_flags & Trace::PF_NoClassName)){
       out << "*I-";
       out.width(10);
       out.flags(std::ios::left);
       out << _currentTraceClassName();
       out << ' ';
     }
     _putStream(out,msg_str);
     break;
   case Trace::Warning:
     flush();
     out << "*W* Code:---------------------------------------------\n";
     out << "*W*   WARNING";
     _putTraceId(out);
     out << ": ";
     _putStream(out,msg_str);
     break;
   case Trace::Error:
     flush();
     out << "*E* Code:---------------------------------------------\n";
     out << "*E*   ERROR";
     _putTraceId(out);
     out << ": ";
     _putStream(out,msg_str);
     break;
   case Trace::Log:
     out << "*L-";
     out.width(8);
     out.flags(std::ios::left);
     out << _currentTraceClassName();
     out << ' ';
     _putStream(out,msg_str);
     break;
   case Trace::Fatal:
     flush();
     out << "*F* Code:---------------------------------------------\n";
     out << "*F*   FATAL";
     _putTraceId(out);
     out << ": ";
     _putStream(out,msg_str);
     break;
   case Trace::ParallelFatal:
     flush();
     out << "*F* Code:---------------------------------------------\n";
     out << "*F*   FATAL";
     _putTraceId(out);
     out << ": ";
     _putStream(out,msg_str);
     break;
   case Trace::Debug:
     out << "*D-";
     out.flags(std::ios::left);
     out.width(10);
     out << _currentTraceClassName();
     out << ' ';
     _putStream(out,msg_str);
     break;
   case Trace::Normal:
     break;
   case Trace::Null:
     out << '\0';
     return;
  }
  out << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// A mettre en correspondance avec Trace::Color
const char* color_fmt[] =
  {
    "30",
    "31", "32", "33", "34", "35", "36", "37",
    "1;31", "1;32", "1;33", "1;34", "1;35", "1;36", "1;37"
  };

void TraceMng::
_writeColor(std::ostream& output,Span<const Byte> input,int color, bool do_flush)
{
  if (color>Trace::Color::LAST_COLOR || color<0)
    color = 0;
  // Pour être sur que les message sont écrits en une seule fois,
  // il ne faut faire qu'un seul write.
  if (color!=0){
    Mutex::ScopedLock sl(m_trace_mutex);
    output << "\33[" << color_fmt[color] << "m";
    Int64 len = input.size();
    // Le message se termine toujours par un '\n'. On écrit la fin de la couleur
    // avant de '\n'
    if (len>0)
      --len;
    _writeSpan(output,input.subspan(0,len));
    output << "\33[0m\n";
  }
  else
    _writeSpan(output,input);

  if (do_flush)
    output.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Écrit le listing.
 *
 * Le listing peut sortir à la fois sur std::cout et dans un
 * ITraceStream si *m_listing_stream* est non nul. Le niveau de verbosité
 * peut être différent dans les deux cas.
 */
void TraceMng::
_writeListing(Span<const Byte> input,Int32 level,int color,bool do_flush)
{
  std::ostream* listing_stream = (m_listing_stream) ? m_listing_stream->stream() : nullptr;
  if (!m_has_color)
    color = 0;

  // Regarde si le niveau de verbosité souhaité est suffisant pour afficher
  // le message.
  Int32 message_level = level;

  // Sortie ITraceStream.
  if (listing_stream){
    Int32 verbosity_level = m_current_class_verbosity_level;
    if (verbosity_level==Trace::UNSPECIFIED_VERBOSITY_LEVEL)
      verbosity_level = m_verbosity_level;
    // Pas de couleur si on redirige les sorties
    int listing_color = 0;
    bool is_printed = (message_level <= verbosity_level);
    if (is_printed)
      _writeColor(*listing_stream,input,listing_color,do_flush);
  }

  // Sortie std::cout
  if (m_is_master || !listing_stream){
    Int32 verbosity_level = m_current_class_verbosity_level;
    if (verbosity_level==Trace::UNSPECIFIED_VERBOSITY_LEVEL)
      verbosity_level = (listing_stream) ? m_stdout_verbosity_level : m_verbosity_level;
    bool is_printed = (message_level <= verbosity_level);
    if (is_printed)
      _writeColor(std::cout,input,color,do_flush);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_write(std::ostream& output,Span<const Byte> input,bool do_flush)
{
  _writeSpan(output,input);
  if (do_flush)
    output.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_write(std::ostream* output,Span<const Byte> input,bool do_flush)
{
  if (!output)
    return;
  _write(*output,input,do_flush);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
beginTrace(const TraceMessage* msg)
{
  Trace::eMessageType id = msg->type();
  if (id==Trace::Null)
    return;
  if (id<Trace::Normal || id>Trace::Null)
    return;
  ++(_getStreamList()->m_str_count[id]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
endTrace(const TraceMessage* msg)
{
  Trace::eMessageType id = msg->type();
  if (id==Trace::Null)
    return;
  if (id<Trace::Normal || id>Trace::Null)
    return;
  Integer n = --(_getStreamList()->m_str_count[id]);
  if (n==0){
    _endTrace(msg);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
putTrace(const String& message,int type)
{
  auto message_type = static_cast<Trace::eMessageType>(type);
  switch(message_type){
  case Trace::Normal:
  case Trace::Info:
    this->info() << message;
    break;
  case Trace::Log:
    this->log() << message;
    break;
  case Trace::Warning:
    this->warning() << message;
    break;
  case Trace::Debug:
    this->debug(Trace::Medium) << message;
    break;
  case Trace::Error:
    this->error() << message;
    break;
  case Trace::Fatal:
    this->fatal() << message;
    break;
  case Trace::ParallelFatal:
    this->pfatal() << message;
    break;
  case Trace::Null:
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_endTrace(const TraceMessage* msg)
{
  Trace::eMessageType id = msg->type();
  TraceMngStreamList* ts = _getStreamList();
  const std::string& str = ts->m_str_list[id]->str();
  const Byte* str_data = reinterpret_cast<const Byte*>(str.data());
  std::vector<Byte> msg_str_copy(str_data,str_data+str.length());
  Span<const Byte> msg_str(msg_str_copy.data(),msg_str_copy.size());

  ts->m_str_list[id]->str(std::string());
  if (msg_str.empty())
    return;

  ts->m_tmp_buf.str(std::string());
  _putTraceMessage(ts->m_tmp_buf,id,msg_str);
  const std::string& tmp_buf_str = ts->m_tmp_buf.str();
  const Byte* tmp_buf_str_data = reinterpret_cast<const Byte*>(tmp_buf_str.data());
  Span<const Byte> buf_array(tmp_buf_str_data,tmp_buf_str.length());

  if (_sendToProxy2(msg,buf_array))
    return;

  _writeDirect(msg,buf_array,msg_str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeDirect(const TraceMessage* msg,Span<const Byte> buf_array,
             Span<const Byte> orig_message)
{
  std::ostream* listing_stream = (m_listing_stream) ? m_listing_stream->stream() : nullptr;
  std::ostream& def_out = (listing_stream) ? (*listing_stream) : std::cout;

  Int32 print_level = -1;
  Trace::eMessageType id = msg->type();
  int color = msg->color();
  // TODO Rendre paramétrable.
  const bool write_stack_trace_for_error = false;
  switch(id){
  case Trace::Normal:
    _writeListing(buf_array,print_level,color,false);
    _checkFlush();
    break;
  case Trace::Info:
    _writeListing(buf_array,msg->level(),color,false);
    _checkFlush();
    break;
  case Trace::Log:
    _write(_logStream(),buf_array);
    _checkFlush();
    break;
  case Trace::Warning:
  case Trace::Error:
    {
      auto error_stream = _errorStream();
      auto log_stream = _logStream();
      auto tc_color = (id==Trace::Warning) ? Trace::Color::DarkYellow : Trace::Color::DarkRed;
      _writeListing(buf_array,print_level,tc_color,true);
      _write(error_stream,buf_array,true);
      String stack_trace = Platform::getStackTrace();
      if (write_stack_trace_for_error)
        _writeStackTrace(&def_out,stack_trace);
      _write(log_stream,buf_array,true);
      _writeStackTrace(log_stream,stack_trace);
      if (listing_stream){
        _write(std::cerr,buf_array);
      }
    }
    break;
  case Trace::Fatal:
  case Trace::ParallelFatal:
    if (m_is_master || id==Trace::Fatal){
      auto error_stream = _errorStream();
      auto log_stream = _logStream();
      _writeListing(buf_array,print_level,Trace::Color::Red,true);
      _write(error_stream,buf_array,true);
      _write(log_stream,buf_array,true);
      String stack_trace = Platform::getStackTrace();
      _writeStackTrace(log_stream,stack_trace);
      if (listing_stream)
        _write(std::cerr,buf_array);
    }
    {
      String s1(orig_message);
      FatalErrorException ex("TraceMng::endTrace()",s1);
      if (id==Trace::ParallelFatal)
        ex.setCollective(true);
      throw ex;
    }
  case Trace::Debug:
    _writeListing(buf_array,print_level,color,true);
    break;
  case Trace::Null:
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
writeDirect(const TraceMessage* msg,const String& str)
{
  Span<const Byte> buf_array(str.bytes());
  _writeDirect(msg,buf_array,buf_array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeStackTrace(std::ostream* output,const String& stack_trace)
{
  if (!output)
    return;
  if (!stack_trace.null()){
    (*output) << "Stack\n" << stack_trace << "\n";
    output->flush();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putStream(std::ostream& ostr,Span<const Byte> buffer)
{
  _putFunctionName(ostr);
  _writeSpan(ostr,buffer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_updateCurrentClassConfig()
{
  m_current_class_verbosity_level = m_current_msg_class.m_info->verboseLevel();
  m_current_class_flags = m_current_msg_class.m_info->flags();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
pushTraceClass(const String& name)
{
  Mutex::ScopedLock sl(m_trace_mutex);
  const TraceClassConfig* tcc = _msgClassConfig(name);
  m_trace_class_stack.push_back(TraceClass(name,tcc));
  m_current_msg_class = m_trace_class_stack.back();
  _updateCurrentClassConfig();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
popTraceClass()
{
  Mutex::ScopedLock sl(m_trace_mutex);
  m_trace_class_stack.pop_back();
  m_current_msg_class = m_trace_class_stack.back();
  _updateCurrentClassConfig();
  flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Termine l'initialisation du gestionnaire de traces.
 */
void TraceMng::
finishInitialize()
{
  m_default_trace_class = TraceClass("Internal",&m_default_trace_class_config);
  m_current_msg_class = m_default_trace_class;
  m_trace_class_stack.push_back(m_default_trace_class);
  if (m_is_master){
    Platform::removeFile(m_error_file_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setClassConfig(const String& name,const TraceClassConfig& config)
{
  Mutex::ScopedLock sl(m_trace_mutex);
  if (name=="*"){
    m_default_trace_class_config = config;
  }
  else{
    auto iter = m_trace_class_config_map.find(name);
    if (iter==m_trace_class_config_map.end()){
      TraceClassConfig* tcc = new TraceClassConfig(config);
      m_trace_class_config_map.insert(std::make_pair(name,tcc));
    }
    else{
      TraceClassConfig* tcc = iter->second;
      *tcc = config;
    }
  }
  // Si la config qui est modifiée correspond à celle en cours il faut la
  // mettre à jour.
  _updateCurrentClassConfig();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceClassConfig TraceMng::
classConfig(const String& name) const
{
  Mutex::ScopedLock sl(m_trace_mutex);
  return *(_msgClassConfig(name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
removeAllClassConfig()
{
  Mutex::ScopedLock sl(m_trace_mutex);
  // Comme tous les TraceClassConfig vont être détruit, il ne faut plus les
  // référencer dans les TraceClass.
  for( auto& i : m_trace_class_stack )
    i.m_info = &m_default_trace_class_config;
  m_current_msg_class.m_info = &m_default_trace_class_config;
  for( const auto& i : m_trace_class_config_map )
    delete i.second;
  m_trace_class_config_map.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
visitClassConfigs(IFunctorWithArgumentT<std::pair<String,TraceClassConfig>>* functor)
{
  if (!functor)
    return;
  for( const auto& i : m_trace_class_config_map ){
    functor->executeFunctor(std::make_pair(i.first,*(i.second)));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setMaster(bool is_master)
{
  m_is_master = is_master;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setVerbosityLevel(Int32 level)
{
  m_verbosity_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setStandardOutputVerbosityLevel(Int32 level)
{
  m_stdout_verbosity_level = level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
resetThreadStatus()
{
  // Détruit et reconstruit le mutex.
  m_trace_mutex->~Mutex();
  new (m_trace_mutex) Mutex();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
addListener(ITraceMessageListener* v)
{
  if (!m_listeners)
    m_listeners = new ListenerList();
  m_listeners->insert(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
removeListener(ITraceMessageListener* v)
{
  if (!m_listeners)
    return;
  m_listeners->erase(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ITraceMng::
fatalMessage(const StandaloneTraceMessage& o)
{
  ARCCORE_FATAL(o.value());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::ITraceMng);
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::ITraceStream);
} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
