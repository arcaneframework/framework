// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HiredisAdapter.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Adapteur pour 'hiredis'.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/std/internal/IRedisContext.h"

#include "arcane_packages.h"

#include <string.h>

#ifdef ARCANE_HAS_PACKAGE_HIREDIS

#include <hiredis/hiredis.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Redis
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HiredisCommand
{
 public:

  ~HiredisCommand()
  {
    _checkDestroy();
  }

  void sendCommand(redisContext* c, const char* format, ...)
  {
    _checkDestroy();
    {
      va_list arguments;
      va_start(arguments, format);
      m_reply = reinterpret_cast<redisReply*>(redisvCommand(c, format, arguments));
      va_end(arguments);
    }
    if (m_reply == nullptr)
      ARCANE_FATAL("Null reply");
    if (c->err != 0)
      ARCANE_FATAL("Error during redis connection: %s\n", c->errstr);
  }
  bool isReplyString() const
  {
    ARCANE_CHECK_POINTER(m_reply);
    return m_reply->type == REDIS_REPLY_STRING;
  }
  size_t replyLength() const
  {
    ARCANE_CHECK_POINTER(m_reply);
    return m_reply->len;
  }
  const char* replyData() const
  {
    ARCANE_CHECK_POINTER(m_reply);
    return m_reply->str;
  }

 private:

  ::redisReply* m_reply = nullptr;

  void _checkDestroy()
  {
    if (m_reply) {
      freeReplyObject(m_reply);
      m_reply = nullptr;
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HiredisContext
: public TraceAccessor
, public IRedisContext
{
 public:

  HiredisContext(ITraceMng* tm)
  : TraceAccessor(tm)
  {}
  ~HiredisContext()
  {
    if (m_redis_context)
      ::redisFree(m_redis_context);
  }

 public:

  void open(const String& machine, Int32 port) override
  {
    redisContext* c = ::redisConnect(machine.localstr(), port);
    m_redis_context = c;
    if (!c)
      ARCANE_FATAL("Error during redis connection: can not allocate redis context");
    if (c->err != 0)
      ARCANE_FATAL("Error during redis connection: {0}", c->errstr);
  }

 public:

  void sendBuffer(const String& key, Span<const std::byte> bytes) override
  {
    _checkContext();
    auto key_bytes = key.bytes();
    size_t key_len = key_bytes.size();
    size_t buf_size = bytes.size();
    HiredisCommand command;
    command.sendCommand(m_redis_context, "SET %b %b", key_bytes.data(), key_len, bytes.data(), buf_size);
  }

  void getBuffer(const String& key, Array<std::byte>& bytes) override
  {
    _checkContext();
    auto key_bytes = key.bytes();
    size_t key_len = key_bytes.size();
    HiredisCommand command;
    command.sendCommand(m_redis_context, "GET %b", key_bytes.data(), key_len);
    if (!command.isReplyString())
      ARCANE_FATAL("Reply is not a string");
    Int64 reply_length = command.replyLength();
    auto* data_as_bytes = reinterpret_cast<const std::byte*>(command.replyData());
    bytes.copy(Span<const std::byte>(data_as_bytes, reply_length));
  }

 public:

  ::redisContext* m_redis_context = nullptr;

 private:

  void _checkContext()
  {
    if (!m_redis_context)
      ARCANE_FATAL("No redis context. You have to call open() to create a context");
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HiredisAdapter
: public TraceAccessor
{
 public:

  HiredisAdapter(ITraceMng* tm)
  : TraceAccessor(tm)
  , m_context(tm)
  {}

 public:

  void test();

 private:

  HiredisContext m_context;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HiredisAdapter::
test()
{
  m_context.open("127.0.0.1", 6379);

  redisContext* c = m_context.m_redis_context;
  char *s, *e;

  HiredisCommand command;

  constexpr const char* REDIS_VERSION_FIELD = "redis_version:";
  command.sendCommand(c, "INFO");
  if (!command.isReplyString())
    ARCANE_FATAL("Can not get INFO");
  if ((s = strstr((char*)command.replyData(), REDIS_VERSION_FIELD)) == NULL)
    ARCANE_FATAL("No INFO");

  s += strlen(REDIS_VERSION_FIELD);

  /* We need a field terminator and at least 'x.y.z' (5) bytes of data */
  if ((e = strstr(s, "\r\n")) == NULL || (e - s) < 5)
    ARCANE_FATAL("Invalid version number");
  info() << "VERSION=" << command.replyData();

  UniqueArray<Int64> my_buffer(10000);
  for (Int64 i = 0, n = my_buffer.size(); i < n; ++i) {
    my_buffer[i] = i + 1;
  }
  Span<const std::byte> send_buf(asBytes(my_buffer));
  Int64 send_size = send_buf.size();
  info() << "SEND_SIZE=" << send_size;
  //command.sendCommand(c, "SET mytest %b", my_buffer.data(), (size_t)send_size);
  m_context.sendBuffer("mytest", send_buf);

  UniqueArray<std::byte> out_bytes;
  m_context.getBuffer("mytest", out_bytes);

  Int64 reply_length = out_bytes.largeSize();
  info() << "REPLY_SIZE = " << reply_length;
  if (reply_length != send_size)
    ARCANE_FATAL("Bad reply v={0} expected={1}", reply_length, send_size);

  Span<const Int64> receive_buf = Arccore::asSpan<Int64>(out_bytes.span());
  if (receive_buf != my_buffer)
    ARCANE_FATAL("Bad value");
  for (Int64 x = 0, n = receive_buf.size(); x < n; ++x)
    if (receive_buf[x] != my_buffer[x])
      ARCANE_FATAL("Bad value i={0} v={1} expected={2}", x, receive_buf[x], my_buffer[x]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Redis

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_HAS_PACKAGE_HIREDIS

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_STD_EXPORT void
_testRedisAdapter([[maybe_unused]] ITraceMng* tm)
{
#ifdef ARCANE_HAS_PACKAGE_HIREDIS
  Redis::HiredisAdapter h{ tm };
  h.test();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Ref<IRedisContext>
createRedisContext([[maybe_unused]] ITraceMng* tm)
{
#ifdef ARCANE_HAS_PACKAGE_HIREDIS
  return makeRef<IRedisContext>(new Redis::HiredisContext(tm));
#else
  ARCANE_FATAL("Can not create Redis context because Arcane is not "
               "compiled with support for 'hiredis' library");
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
