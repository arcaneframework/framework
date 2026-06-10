// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITraceMng.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Trace manager.                                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_ITRACEMNG_H
#define ARCCORE_TRACE_ITRACEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/trace/TraceMessage.h"
#include "arccore/base/RefDeclarations.h"

#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Arguments for ITraceMessageListener::visitMessage().
 *
 * \a buffer() contains the character string to display.
 * \a buffer() always ends with a null terminator.
 *
 * An instance of this class is a temporary object that should not
 * be kept beyond the call to ITraceMessageListener::visitMessage().
 */
class ARCCORE_TRACE_EXPORT TraceMessageListenerArgs
{
 public:

  TraceMessageListenerArgs(const TraceMessage* msg, ConstArrayView<char> buf)
  : m_message(msg)
  , m_buffer(buf)
  {}

 public:

  //! Trace message information
  const TraceMessage* message() const
  {
    return m_message;
  }

  //! Message character string.
  ConstArrayView<char> buffer() const
  {
    return m_buffer;
  }

 private:

  const TraceMessage* m_message;
  ConstArrayView<char> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a visitor for trace messages.
 */
class ARCCORE_TRACE_EXPORT ITraceMessageListener
{
 public:

  virtual ~ITraceMessageListener() {}
  /*!
   * \brief Receiving message \a msg containing string \a str.
   *
   * If the return value is \a true, the message is not used by ITraceMng.
   *
   * The instance has the right to call ITraceMng::writeDirect() to write
   * messages directly during the call to this method.
   *
   * \warning Attention, this function must be thread-safe because it can be
   * called simultaneously by multiple threads.
   */
  virtual bool visitMessage(const TraceMessageListenerArgs& args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Stream for a trace.
 *
 * This instance uses a reference counter and can be manipulated
 * via a ReferenceCounter instance.
 */
class ARCCORE_TRACE_EXPORT ITraceStream
{
 public:

  typedef ReferenceCounterTag ReferenceCounterTagType;

 public:

  virtual ~ITraceStream() = default;

 public:

  //! Adds a reference.
  virtual void addReference() = 0;
  //! Removes a reference.
  virtual void removeReference() = 0;
  //! Associated standard stream. May return null.
  virtual std::ostream* stream() = 0;

 public:

  static ITraceStream* createFileStream(const String& filename);
  static ITraceStream* createStream(std::ostream* stream, bool need_destroy);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Trace manager interface.
 *
 * An instance of this class manages trace streams.
 * To send a message, simply call the corresponding method
 * (info() for an information message, error() for an error, ...)
 * to retrieve a stream and use the << operator on this stream to
 * transmit a message.
 *
 * For example:
 * \code
 * ITraceMng* tr = ...;
 * tr->info() << "This is an information message.";
 * int proc_id = 0;
 * tr->error() << "Error on processor " << proc_id;
 * \endcode
 *
 * The message is sent upon destruction of the stream. In the
 * previous examples, the streams are temporarily created (by the info() method)
 * and destroyed as soon as the << operator has been applied to them.
 *
 * \warning It is absolutely necessary to call the finishInitialize() method before
 * using the calls to the pushTraceClass() and popTraceClass() methods.
 *
 * If you want to send a message in several parts, you must store
 * the returned stream:
 *
 * \code
 * TraceMessage info = m_trace_mng->info();
 * info() << "Start of information.\n"
 * info() << "End of information.";
 * \endcode
 *
 * It is possible to use simple formatters on messages
 * (via the #TraceMessage class)
 * or standard iostream formatters by applying the operator() operator
 * of TraceMessage.
 *
 * Instances of this class are managed by a reference counter. It
 * is preferable to keep instances in a ReferenceCounter.
 */
class ARCCORE_TRACE_EXPORT ITraceMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~ITraceMng() = default;

 public:

  //! Stream for an error message
  virtual TraceMessage error() = 0;
  //! Stream for a parallel error message
  virtual TraceMessage perror() = 0;
  //! Stream for a fatal error message
  virtual TraceMessage fatal() = 0;
  //! Stream for a parallel fatal error message
  virtual TraceMessage pfatal() = 0;
  //! Stream for a warning message
  virtual TraceMessage warning() = 0;
  //! Stream for a parallel warning message
  virtual TraceMessage pwarning() = 0;
  //! Stream for an information message
  virtual TraceMessage info() = 0;
  //! Stream for a parallel information message
  virtual TraceMessage pinfo() = 0;
  //! Stream for an information message of a given category
  virtual TraceMessage info(char category) = 0;
  //! Stream for an information message of a given level
  virtual TraceMessage info(Int32 level) = 0;
  //! Stream for a parallel information message of a given category
  virtual TraceMessage pinfo(char category) = 0;
  //! Stream for a conditional information message.
  virtual TraceMessage info(bool) = 0;
  //! Stream for a log message.
  virtual TraceMessage log() = 0;
  //! Stream for a parallel log message.
  virtual TraceMessage plog() = 0;
  //! Stream for a log message preceded by the time.
  virtual TraceMessage logdate() = 0;
  //! Stream for a debug message.
  virtual TraceMessageDbg debug(Trace::eDebugLevel = Trace::Medium) = 0;
  //! Stream for an unused message
  virtual TraceMessage devNull() = 0;

  //! \deprecated Use setInfoActivated() instead
  ARCCORE_DEPRECATED_2018 virtual bool setActivated(bool v) { return setInfoActivated(v); }
  /*!
   * \brief Modifies the activation state of info messages.
   *
   * \return the previous activation state.
   */
  virtual bool setInfoActivated(bool v) = 0;
  //! Indicates if info message outputs are activated.
  virtual bool isInfoActivated() const = 0;

  //! Finishes initialization
  virtual void finishInitialize() = 0;

  /*!
   * \brief Adds class \a s to the stack of active message classes.
   * \threadsafe
   */
  virtual void pushTraceClass(const String& name) = 0;

  /*!
   * \brief Removes the last message class from the stack.
   * \threadsafe
   */
  virtual void popTraceClass() = 0;

  //! Flushes all streams.
  virtual void flush() = 0;

  /*!
   * \brief Redirects all messages to the stream \a o.
   * \deprecated Use the setRedirectStream(ITraceStream*) overload.
   */
  ARCCORE_DEPRECATED_2018 virtual void setRedirectStream(std::ostream* o) = 0;

  //! Redirects all messages to the stream \a o
  virtual void setRedirectStream(ITraceStream* o) = 0;

  //! Returns the dbg level of the configuration file
  virtual Trace::eDebugLevel configDbgLevel() const = 0;

 public:

  /*!
   * \brief Adds observer \a v to this message manager.
   *
   * The caller remains the owner of \a v and must remove it
   * via removeListener() before destroying it.
   */
  virtual void addListener(ITraceMessageListener* v) = 0;

  //! Removes observer \a v from this message manager.
  virtual void removeListener(ITraceMessageListener* v) = 0;

  /*!
   * \brief Sets the manager identifier.
   *
   * If not null, the identifier is displayed in case of an error to
   * identify the instance displaying the message. The identifier
   * can be arbitrary. By default, it is the process rank and
   * the machine name.
   */
  virtual void setTraceId(const String& id) = 0;

  //! Manager identifier.
  virtual const String& traceId() const = 0;

  /*!
   * \brief Sets the error file name to \a file_name.
   *
   * If an error file is already open, it is closed and a new one
   * with this new file name will be created upon the next error.
   *
   * If \a file_name is the null string, no error file is used.
   */
  virtual void setErrorFileName(const String& file_name) = 0;

  /*!
   * \brief Sets the log file name to \a file_name.
   *
   * If a log file is already open, it is closed and a new one
   * with this new file name will be created upon the next log.
   *
   * If \a file_name is the null string, no log file is used.
   */
  virtual void setLogFileName(const String& file_name) = 0;

 public:

  //! Signals the start of writing message \a message
  virtual void beginTrace(const TraceMessage* message) = 0;

  //! Signals the end of writing message \a message
  virtual void endTrace(const TraceMessage* message) = 0;

  /*!
   * \brief Directly sends a message of type \a type.
   *
   * \a type must correspond to Trace::eMessageType.
   * This method should only be used by the .NET wrapping.
   */
  virtual void putTrace(const String& message, int type) = 0;

 public:

  //! Sets the configuration for the message class \a name
  virtual void setClassConfig(const String& name, const TraceClassConfig& config) = 0;

  //! Configuration associated with the message class \a name
  virtual TraceClassConfig classConfig(const String& name) const = 0;

  /*!
   * \brief Sets the 'master' state of the instance.
   *
   * Instances that have this attribute set to \a true display
   * messages on std::cout as well as the messages
   * perror() and pwarning(). It is therefore preferable that there be
   * only one master ITraceMng instance.
   */
  virtual void setMaster(bool is_master) = 0;

  // Indicates if the instance is master.
  virtual bool isMaster() const = 0;

  /*!
   * \brief Sets the verbosity level of the outputs.
   *
   * Messages at a level higher than this level are not outputted.
   * The level used is the one given as an argument to info(Int32).
   * The default level is the one given by TraceMessage::DEFAULT_LEVEL.
   */
  virtual void setVerbosityLevel(Int32 level) = 0;

  //! Message verbosity level
  virtual Int32 verbosityLevel() const = 0;

  /*!
   * \brief Sets the verbosity level of outputs on std::cout
   *
   * This property is only used if isMaster() is true and
   * if the listings outputs have been redirected. Otherwise, the property
   * verbosityLevel() is used.
   */
  virtual void setStandardOutputVerbosityLevel(Int32 level) = 0;

  //! Message verbosity level on std::cout
  virtual Int32 standardOutputVerbosityLevel() const = 0;

  /*!
   * \internal
   * Indicates that the thread manager has changed and that the structures
   * managing multi-threading must be re-declared.
   * Internal to Arccore, do not use.
   */
  virtual void resetThreadStatus() = 0;

  /*!
   * \brief Writes a message directly.
   *
   * Directly writes message \a msg containing string \a buf_array.
   * The message is not analyzed by the instance and is always written
   * without any specific formatting. This operation should in principle
   * only be used by an ITraceMessageListener. For other cases,
   * standard traces must be used.
   */
  virtual void writeDirect(const TraceMessage* msg, const String& str) = 0;

  //! Removes all configuration classes set via setClassConfig().
  virtual void removeAllClassConfig() = 0;

  /*!
   * \biref Applies the functor \a functor to all registered TraceClassConfig.
   *
   * The first argument of the pair is the configuration class name and
   * the second is its value as returned by classConfig().
   *
   * It is permitted to modify the TraceClassConfig during visitation via
   * a call to setClassConfig().
   */
  virtual void visitClassConfigs(IFunctorWithArgumentT<std::pair<String, TraceClassConfig>>* functor) = 0;

 public:

  /*!
   * \brief Performs a fatal() on an already manufactured message.
   *
   * This method allows writing code equivalent to:
   *
   * \code
   * fatal() << "MyMessage";
   * \endcode
   *
   * like this:
   *
   * \code
   * fatalMessage(StandaloneTraceMessage{} << "MyMessage");
   * \endcode
   *
   * This second solution allows signaling to the compiler that
   * the method will not return and thus avoid certain compilation warnings.
   */
  void fatalMessage [[noreturn]] (const StandaloneTraceMessage& o);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_TRACE_EXPORT ITraceMng* arccoreCreateDefaultTraceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
using Arcane::arccoreCreateDefaultTraceMng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
