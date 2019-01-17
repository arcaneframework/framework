// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* TraceMng.cc                                                 (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire des traces.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/ITraceMng.h"
#include "arccore/trace/TraceClassConfig.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/IFunctor.h"
#include "arccore/base/StackTrace.h"
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/String.h"
#include "arccore/base/PlatformUtils.h"

#include "arccore/concurrency/Mutex.h"

#include "arccore/concurrency/ThreadPrivate.h"

#include <sstream>
#include <fstream>
#include <limits>
#include <set>
#include <map>
#include <vector>

#include <glib.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

// TODO réimplémenter cette classe
class TraceTimer
{
 public:
  double getTime() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TraceStream
{
 public:
  // A mettre en correspondance avec Trace::Trace::eMessageType
  static const Integer NB_STREAM = 9;
 public:
  void build()
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
 * \internal
 * \brief Implémentation du gestionnaire de traces.
 */
class TraceMng
: public ITraceMng
{
 public:

 public:

  TraceMng();
  virtual ~TraceMng();

 public:
	
  virtual TraceMessage operator()()
    { return TraceMessage(_getStream(Trace::Normal),this,Trace::Normal); }
  virtual TraceMessage info()
    { return (_isCurrentClassActivated()) ? _info() : _devNull(); }
  virtual TraceMessage info(char /*category*/)
    { return (_isCurrentClassActivated()) ? _info() : _devNull(); }
  virtual TraceMessage info(Int32 verbose_level)
    { return (_isCurrentClassActivated()) ? _info(verbose_level) : _devNull(); }
  virtual TraceMessage pinfo()
    { return (_isCurrentClassParallelActivated()) ? _info() : _devNull(); }
  virtual TraceMessage pinfo(char /*category*/)
    { return (_isCurrentClassParallelActivated()) ? _info() : _devNull(); }
  virtual TraceMessage info(bool is_ok)
    { return (is_ok) ? _info() : _devNull(); }
  virtual TraceMessage warning()
    { return TraceMessage(_getStream(Trace::Warning),this,Trace::Warning); }
  virtual TraceMessage pwarning()
    { return m_is_master ? warning() : _devNull(); }
  virtual TraceMessage error()
    { return TraceMessage(_getStream(Trace::Error),this,Trace::Error); }
  virtual TraceMessage perror()
    { return m_is_master ? error() : _devNull(); }
  virtual TraceMessage log()
    { return (_isCurrentClassActivated()) ? _log(false) : _devNull(); }
  virtual TraceMessage plog()
    { return (_isCurrentClassParallelActivated()) ? _log(false) : _devNull(); }
  virtual TraceMessage logdate()
    { return (_isCurrentClassActivated()) ? _log(true) : _devNull(); }
  virtual TraceMessage fatal()
    { return TraceMessage(_getStream(Trace::Fatal),this,Trace::Fatal); }
  virtual TraceMessage pfatal()
    { return TraceMessage(_getStream(Trace::ParallelFatal),this,Trace::ParallelFatal); }
  virtual TraceMessage devNull()
    { return _devNull(); }
  virtual TraceMessageDbg debug(Trace::eDebugLevel dbg_lvl)
    { return (dbg_lvl<=_configDbgLevel()) ? _dbg() : _dbgDevNull(); }
  virtual void endTrace(const TraceMessage* msg);
  virtual void beginTrace(const TraceMessage* msg);
  virtual void putTrace(const String& message,int type);
	
  virtual void addListener(ITraceMessageListener* v);
  virtual void removeListener(ITraceMessageListener* v);

  virtual bool setActivated(bool v)
    { bool old = m_is_activated; m_is_activated = v; return old; }

  virtual void finishInitialize();

  virtual void pushTraceClass(const String& s);
  virtual void popTraceClass();

  virtual void flush();

  virtual void setRedirectStream(std::ostream* ro)
  { m_redirect_stream = ro; }

  virtual Trace::eDebugLevel configDbgLevel() const
  { return _configDbgLevel(); }

  virtual void setErrorFileName(const String& file_name);
  virtual void setLogFileName(const String& file_name);

  virtual void setClassConfig(const String& name,const TraceClassConfig& config);
  virtual TraceClassConfig classConfig(const String& name) const;
  virtual void removeAllClassConfig();

  virtual void setMaster(bool is_master);

  virtual bool isMaster() const
  { return m_is_master; }

  virtual void setVerbosityLevel(Int32 level);
  virtual Int32 verbosityLevel() const
  {
    return m_verbosity_level;
  }

  virtual void resetThreadStatus();

  virtual void writeDirect(const TraceMessage* msg,const String& str);

  virtual void setTraceId(const String& id) { m_trace_id = id; }
  virtual const String& traceId() const { return m_trace_id; }

  virtual void visitClassConfigs(IFunctorWithArgumentT<std::pair<String,TraceClassConfig>>* functor);

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
    { return TraceMessage(_getStream(Trace::Null),this,Trace::Null); }
#ifdef ARCCORE_DEBUG
  TraceMessageDbg _dbg()
    { return TraceMessage(_getStream(Trace::Debug),this,Trace::Debug); }
  TraceMessageDbg _dbgDevNull()
    { return TraceMessage(_getStream(Trace::Null),this,Trace::Null); }
#else
  TraceMessageDbg _dbg()
    { return TraceMessageDbg(); }
  TraceMessageDbg _dbgDevNull()
    { return TraceMessageDbg(); }
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

  static ThreadPrivateStorage m_string_list_key;

  bool m_is_master;
  bool m_want_trace_function;
  bool m_want_trace_timer;
  Int32 m_verbosity_level;
  Int32 m_current_class_verbosity_level;
  Int32 m_current_class_flags;
  ThreadPrivate<TraceStream> m_strs;
  ListenerList* m_listeners;
  bool m_is_activated;
  std::map<String,TraceClassConfig*> m_trace_class_config_map;
  TraceClassStack m_trace_class_stack;
  TraceClassConfig m_default_trace_class_config;
  TraceClass m_default_trace_class;
  TraceClass m_current_msg_class;
  std::ostream* m_redirect_stream;
  Integer m_nb_flush;
  String m_error_file_name;
  String m_log_file_name;
  String m_trace_id;
  std::ofstream* m_error_file;
  std::ofstream* m_log_file;
  Mutex* m_trace_mutex;
  bool m_is_error_disabled;
  bool m_is_log_disabled;
  bool m_has_color;

  TraceTimer m_trace_timer;
  void _writeTimeString(std::ostream& out);

 private:

  std::ostream* _getStream(Trace::eMessageType id)
  {
    int iid = static_cast<int>(id);
    std::ostringstream* ostr = m_strs.item()->m_str_list[iid];
    ostr->str(std::string());
    return ostr;
  }
  bool _sendToProxy2(const TraceMessage* msg,ConstArrayView<char> str);

  //NOTE: cette méthode doit être appelée avec le verrou \a m_trace_mutex positionné.
  const TraceClassConfig* _msgClassConfig(const String& s) const
  {
    auto ci = m_trace_class_config_map.find(s);
    if (ci!=m_trace_class_config_map.end()){
      return ci->second;
    }
    return &m_default_trace_class_config;
  }
  const String& _currentTraceClassName() const
    { return m_current_msg_class.m_name; }
  void _checkFlush();
  void _putStream(std::ostream& ostr,ConstArrayView<char> buffer);
  void _putTraceMessage(std::ostream& ostr,Trace::eMessageType id,ConstArrayView<char>);
  void _putDate(std::ostream& ostr);
  std::ostream* _errorFile();
  std::ostream* _logFile();
  void _write(std::ostream& output,ConstArrayView<char> input,bool do_flush=false);
  void _writeColor(std::ostream& output,ConstArrayView<char> input,int color,bool do_flush);
  void _write(std::ostream* output,ConstArrayView<char> input,bool do_flush=false);
  void _writeStackTrace(std::ostream* output);
  void _endTrace(const TraceMessage* msg);
  void _putFunctionName(std::ostream& out);
  void _writeDirect(const TraceMessage* msg,ConstArrayView<char> buf_array,
                    ConstArrayView<char> orig_message);
  void _putTraceId(std::ostream& out);
  void _updateCurrentClassConfig();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ThreadPrivateStorage TraceMng::m_string_list_key;

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
: m_is_master(true)
, m_want_trace_function(false)
, m_want_trace_timer(false)
, m_verbosity_level(TraceMessage::DEFAULT_LEVEL)
, m_current_class_verbosity_level(TraceMessage::DEFAULT_LEVEL)
, m_current_class_flags(Trace::PF_Default)
, m_strs(&m_string_list_key)
, m_listeners(nullptr)
, m_is_activated(true)
, m_default_trace_class("Internal",&m_default_trace_class_config)
, m_current_msg_class(m_default_trace_class)
, m_redirect_stream(nullptr)
, m_nb_flush(0)
, m_error_file_name("errors")
, m_error_file(nullptr)
, m_log_file(nullptr)
, m_trace_mutex(new Mutex())
, m_is_error_disabled(false)
, m_is_log_disabled(false)
, m_has_color(false)
{
  // La première instance de cette classe est créée via
  // la classe Application et il y a nécessairement qu'un seul
  // thread qui fasse cet appel. Il n'y a donc pas besoin
  // de protéger la création de cette clé privée.
  // A noter qu'à partir de la GLib 2.32, il n'y a plus besoin de créer
  // explicitement la partie privée.
  m_string_list_key.initialize();

#if OLD
  {
    String s;

    s = Platform::getEnvironmentVariable("ARCCORE_TRACE_FUNCTION");
    if (s=="TRUE" || s=="1")
      m_want_trace_function = true;

    s = platform::getEnvironmentVariable("ARCCORE_TRACE_TIMER");
    if (s=="TRUE" || s=="1")
      m_want_trace_timer = true;
  }
#endif

  m_has_color = Platform::getConsoleHasColor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceMng::
~TraceMng()
{
  for( auto i : m_trace_class_config_map )
    delete i.second;
  if (m_error_file)
    m_error_file->close();
  if (m_log_file)
    m_log_file->close();
  delete m_error_file;
  delete m_log_file;
  delete m_listeners;
  delete m_trace_mutex;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool TraceMng::
_sendToProxy2(const TraceMessage* msg,ConstArrayView<char> buf)
{
  if (m_listeners){
    TraceMessageListenerArgs args(msg,buf);
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
_errorFile()
{
  if (m_is_error_disabled)
    return 0;
  if (!m_error_file)
    m_error_file = new std::ofstream(m_error_file_name.localstr());
  return m_error_file;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream* TraceMng::
_logFile()
{
  if (m_is_log_disabled)
    return 0;
  if (!m_log_file){
    if (m_log_file_name.null()){
      m_is_log_disabled = true;
      return 0;
    }
    m_log_file = new std::ofstream(m_log_file_name.localstr());
  }
  return m_log_file;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
flush()
{
  std::cout.flush();
  if (m_redirect_stream)
    m_redirect_stream->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setErrorFileName(const String& file_name)
{
  if (m_error_file_name==file_name)
    return;
  m_error_file_name = file_name;
  if (m_error_file){
    m_error_file->close();
    delete m_error_file;
    m_error_file = 0;
  }
  m_is_error_disabled = m_log_file_name.null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
setLogFileName(const String& file_name)
{
  if (m_log_file_name==file_name)
    return;
  m_log_file_name = file_name;
  if (m_log_file){
    m_log_file->close();
    delete m_log_file;
    m_log_file = 0;
  }
  m_is_log_disabled = m_log_file_name.null();
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
_checkFlush()
{
  std::cout.flush();
  ++m_nb_flush;
  if ( (m_nb_flush % 50) == 0 ){
    m_nb_flush = 0;
    if (m_redirect_stream)
      m_redirect_stream->flush();
    std::cout.flush();
  }
  std::cout.flush();
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
_putTraceMessage(std::ostream& out,Trace::eMessageType id,ConstArrayView<char> msg_str)
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
_writeColor(std::ostream& output,ConstArrayView<char> input,int color, bool do_flush)
{
  if (color>Trace::Color::LAST_COLOR || color<0)
    color = 0;
  // Pour être sur que les message sont écrits en une seule fois,
  // il ne faut faire qu'un seul write.
  if (color!=0){
    Mutex::ScopedLock sl(m_trace_mutex);
    output << "\33[" << color_fmt[color] << "m";
    Integer len = input.size();
    // Le message se termine toujours par un '\n'. On écrit la fin de la couleur
    // avant de '\n'
    if (len>0)
      --len;
    output.write((char*)input.data(),len);
    output << "\33[0m\n";
  }
  else
    output.write((char*)input.data(),input.size());

  if (do_flush)
    output.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_write(std::ostream& output,ConstArrayView<char> input,bool do_flush)
{
  output.write((char*)input.data(),input.size());
  //output << '\n';
  if (do_flush)
    output.flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_write(std::ostream* output,ConstArrayView<char> input,bool do_flush)
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
  ++m_strs.item()->m_str_count[id];
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
  Integer n = --m_strs.item()->m_str_count[id];
  if (n==0){
    _endTrace(msg);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
putTrace(const String& message,int type)
{
  Trace::eMessageType message_type = (Trace::eMessageType)type;
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
  TraceStream* ts = m_strs.item();
  const std::string& str = ts->m_str_list[id]->str();
  Integer str_len = arccoreCheckArraySize(str.length());
  ConstArrayView<char> c_array(str_len,str.c_str());
  std::vector<char> msg_str_copy(c_array.range().begin(),c_array.range().end());
  ConstArrayView<char> msg_str(arccoreCheckArraySize(msg_str_copy.size()),msg_str_copy.data());

  ts->m_str_list[id]->str(std::string());
  if (msg_str.empty())
    return;

  ts->m_tmp_buf.str(std::string());
  _putTraceMessage(ts->m_tmp_buf,id,msg_str);
  const std::string& tmp_buf_str = ts->m_tmp_buf.str();
  Integer tmp_buf_len = arccoreCheckArraySize(tmp_buf_str.length());
  ConstArrayView<char> buf_array(tmp_buf_len,tmp_buf_str.c_str());

  if (_sendToProxy2(msg,buf_array))
    return;

  _writeDirect(msg,buf_array,msg_str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeDirect(const TraceMessage* msg,ConstArrayView<char> buf_array,
             ConstArrayView<char> orig_message)
{
  std::ostream& def_out = (m_redirect_stream) ? (*m_redirect_stream) : std::cout;
  std::ostream& def_err = (m_redirect_stream) ? (*m_redirect_stream) : std::cerr;

  // Regarde si le niveau de verbosité souhaité est suffisant pour afficher
  // le message.
  Int32 verbosity_level = m_current_class_verbosity_level;
  if (verbosity_level==Trace::UNSPECIFIED_VERBOSITY_LEVEL)
    verbosity_level = m_verbosity_level;
  bool is_printed = msg->level() <= verbosity_level;

  Trace::eMessageType id = msg->type();
  int color = msg->color();
  // TODO Rendre paramétrable.
  const bool write_stack_trace_for_error = false;
  // Pas de couleur si on redirige les sorties
  if (m_redirect_stream || !m_has_color)
    color = 0;
  switch(id){
  case Trace::Normal:
    _writeColor(def_out,buf_array,color,false);
    _checkFlush();
    break;
  case Trace::Info:
    if (is_printed){
      _writeColor(def_out,buf_array,color,false);
      _checkFlush();
    }
    break;
  case Trace::Log:
    _write(_logFile(),buf_array);
    _checkFlush();
    break;
  case Trace::Warning:
    _writeColor(def_out,buf_array,color!=0 ? color : Trace::Color::DarkYellow,true);
    _write(_errorFile(),buf_array,true);
    if (write_stack_trace_for_error)
      _writeStackTrace(&def_out);
    _write(_logFile(),buf_array,true);
    _writeStackTrace(_logFile());
    if (&def_err!=&std::cerr)
      _write(std::cerr,buf_array);
    break;
  case Trace::Error:
    _writeColor(def_out,buf_array,color!=0 ? color : Trace::Color::DarkRed,true);
    _write(_errorFile(),buf_array,true);
    if (write_stack_trace_for_error)
      _writeStackTrace(&def_out);
    _write(_logFile(),buf_array,true);
    _writeStackTrace(_logFile());
    if (&def_err!=&std::cerr)
      _write(std::cerr,buf_array);
    break;
  case Trace::Fatal:
  case Trace::ParallelFatal:
    if (m_is_master || id==Trace::Fatal){
      _writeColor(def_out,buf_array,Trace::Color::Red,true);
      _write(_errorFile(),buf_array,true);
      _write(_logFile(),buf_array,true);
      _writeStackTrace(_logFile());
      if (&def_err!=&std::cerr)
        _write(std::cerr,buf_array);
    }
    {
      String s1(orig_message.data(),orig_message.size());
      FatalErrorException ex("TraceMng::endTrace()",s1);
      if (id==Trace::Fatal)
        throw ex;
      if (id==Trace::ParallelFatal){
        ex.setCollective(true);
        throw ex;
      }
    }
    break;
  case Trace::Debug:
    _writeColor(def_out,buf_array,color,true);
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
  ConstArrayView<char> buf_array(str.len()+1,str.localstr());
  _writeDirect(msg,buf_array,buf_array);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_writeStackTrace(std::ostream* output)
{
  if (!output)
    return;
  IStackTraceService* stack_service = Platform::getStackTraceService();
  if (stack_service){
    StackTrace stack_trace = stack_service->stackTrace();
    (*output) << "Stack\n" << stack_trace.toString() << "\n";
    output->flush();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMng::
_putStream(std::ostream& ostr,ConstArrayView<char> buffer)
{
  _putFunctionName(ostr);
  ostr.write(buffer.data(),buffer.size());
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
  for( auto i : m_trace_class_stack )
    i.m_info = &m_default_trace_class_config;
  m_current_msg_class.m_info = &m_default_trace_class_config;
  for( auto i : m_trace_class_config_map )
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
  for( auto i : m_trace_class_config_map ){
    std::pair<String,TraceClassConfig> x(i.first,*(i.second));
    functor->executeFunctor(x);
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
