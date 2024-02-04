// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SHA3HashAlgorithm.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Calcule de fonction de hashage SHA-3.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/SHA3HashAlgorithm.h"

#include <cstring>

// L'algorithme est decrit ici;
// https://en.wikipedia.org/wiki/SHA-3

// L'implémentation est issue du dépot suivant:
// https : //github.com/rhash/RHash

/* sha3.c - an implementation of Secure Hash Algorithm 3 (Keccak).
 * based on the
 * The Keccak SHA-3 submission. Submission to NIST (Round 3), 2011
 * by Guido Bertoni, Joan Daemen, Michaël Peeters and Gilles Van Assche
 *
 * Copyright (c) 2013, Aleksey Kravchenko <rhash.admin@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE  INCLUDING ALL IMPLIED WARRANTIES OF  MERCHANTABILITY
 * AND FITNESS.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT,  OR CONSEQUENTIAL DAMAGES  OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE,  DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,  NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF  OR IN CONNECTION  WITH THE USE  OR
 * PERFORMANCE OF THIS SOFTWARE.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

namespace Arcane::SHA3Algorithm
{

namespace
{
  // constants
  constexpr int sha3_max_permutation_size = 25;
  constexpr int sha3_max_rate_in_qwords = 24;

  constexpr int NumberOfRounds = 24;

  //! SHA3 (Keccak) constants for 24 rounds
  constexpr uint64_t keccak_round_constants[NumberOfRounds] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
  };

  // ROTL/ROTR rotate a 64-bit word left by n bits
  static uint64_t _rotateLeft(uint64_t qword, int n)
  {
    return ((qword) << (n) ^ ((qword) >> (64 - (n))));
  }
} // namespace

class SHA3
{

  //! Algorithm context.
  struct sha3_ctx
  {
    /* 1600 bits algorithm hashing state */
    uint64_t hash[sha3_max_permutation_size];
    /* 1536-bit buffer for leftovers */
    uint64_t message[sha3_max_rate_in_qwords];
    /* count of bytes in the message[] buffer */
    unsigned int rest;
    /* size of a message block processed at once */
    unsigned int block_size;
  };
  sha3_ctx m_context;

 public:

  void keccak_init(unsigned int bits);
  void sha3_224_init();
  void sha3_256_init();
  void sha3_384_init();
  void sha3_512_init();

  static void keccak_theta(uint64_t* A);
  static void keccak_pi(uint64_t* A);
  static void keccak_chi(uint64_t* A);
  static void sha3_permutation(uint64_t* state);
  static void sha3_process_block(uint64_t hash[25], const uint64_t* block, size_t block_size);
  void sha3_update(Span<const std::byte> bytes);
  void sha3_final(ByteArray& output_hash);
};

// Valide pour little-endian
//@{
#define le2me_64(x) (x)
#define me64_to_le_str(to, from, length) std::memcpy((to), (from), (length))
//@}

#define IS_ALIGNED_64(p) (0 == (7 & (uintptr_t)(p)))

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Initializing a sha3 context for given number of output bits
void SHA3::
keccak_init(unsigned int bits)
{
  /* NB: The Keccak capacity parameter = bits * 2 */
  unsigned rate = 1600 - bits * 2;

  std::memset(&m_context, 0, sizeof(sha3_ctx));
  m_context.block_size = rate / 8;
  bool is_ok = (rate <= 1600 && (rate % 64) == 0);
  if (!is_ok)
    ARCANE_FATAL("Bad value for rate '{0}'", rate);
  // La taille de bloc est au maximum de 144 pour SHA3-224
  // Au dela, la fonction 'sha3_process_block' ne fonctionne pas
  if (m_context.block_size > 144)
    ARCANE_FATAL("Block size is too big (v={0}) max_allowed=144", m_context.block_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialize context before calculating hash.
 */
void SHA3::
sha3_224_init()
{
  keccak_init(224);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialize context before calculating hash.
 */
void SHA3::
sha3_256_init()
{
  keccak_init(256);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialize context before calculating hash.
 */
void SHA3::
sha3_384_init()
{
  keccak_init(384);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialize context before calculating hash.
 */
void SHA3::
sha3_512_init()
{
  keccak_init(512);
}

#define XORED_A(i) A[(i)] ^ A[(i) + 5] ^ A[(i) + 10] ^ A[(i) + 15] ^ A[(i) + 20]
#define THETA_STEP(i) \
  A[(i)] ^= D[(i)]; \
  A[(i) + 5] ^= D[(i)]; \
  A[(i) + 10] ^= D[(i)]; \
  A[(i) + 15] ^= D[(i)]; \
  A[(i) + 20] ^= D[(i)]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Keccak theta() transformation
void SHA3::
keccak_theta(uint64_t* A)
{
  uint64_t D[5];
  D[0] = _rotateLeft(XORED_A(1), 1) ^ XORED_A(4);
  D[1] = _rotateLeft(XORED_A(2), 1) ^ XORED_A(0);
  D[2] = _rotateLeft(XORED_A(3), 1) ^ XORED_A(1);
  D[3] = _rotateLeft(XORED_A(4), 1) ^ XORED_A(2);
  D[4] = _rotateLeft(XORED_A(0), 1) ^ XORED_A(3);
  THETA_STEP(0);
  THETA_STEP(1);
  THETA_STEP(2);
  THETA_STEP(3);
  THETA_STEP(4);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Keccak pi() transformation
void SHA3::
keccak_pi(uint64_t* A)
{
  uint64_t A1;
  A1 = A[1];
  A[1] = A[6];
  A[6] = A[9];
  A[9] = A[22];
  A[22] = A[14];
  A[14] = A[20];
  A[20] = A[2];
  A[2] = A[12];
  A[12] = A[13];
  A[13] = A[19];
  A[19] = A[23];
  A[23] = A[15];
  A[15] = A[4];
  A[4] = A[24];
  A[24] = A[21];
  A[21] = A[8];
  A[8] = A[16];
  A[16] = A[5];
  A[5] = A[3];
  A[3] = A[18];
  A[18] = A[17];
  A[17] = A[11];
  A[11] = A[7];
  A[7] = A[10];
  A[10] = A1;
  // note: A[ 0] is left as is
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define CHI_STEP(i)                             \
  A0 = A[0 + (i)]; \
  A1 = A[1 + (i)]; \
  A[0 + (i)] ^= ~A1 & A[2 + (i)]; \
  A[1 + (i)] ^= ~A[2 + (i)] & A[3 + (i)]; \
  A[2 + (i)] ^= ~A[3 + (i)] & A[4 + (i)]; \
  A[3 + (i)] ^= ~A[4 + (i)] & A0; \
  A[4 + (i)] ^= ~A0 & A1

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Keccak chi() transformation
void SHA3::
keccak_chi(uint64_t* A)
{
  uint64_t A0, A1;
  CHI_STEP(0);
  CHI_STEP(5);
  CHI_STEP(10);
  CHI_STEP(15);
  CHI_STEP(20);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3::
sha3_permutation(uint64_t* state)
{
  int round;
  for (round = 0; round < NumberOfRounds; round++) {
    keccak_theta(state);

    /* apply Keccak rho() transformation */
    state[1] = _rotateLeft(state[1], 1);
    state[2] = _rotateLeft(state[2], 62);
    state[3] = _rotateLeft(state[3], 28);
    state[4] = _rotateLeft(state[4], 27);
    state[5] = _rotateLeft(state[5], 36);
    state[6] = _rotateLeft(state[6], 44);
    state[7] = _rotateLeft(state[7], 6);
    state[8] = _rotateLeft(state[8], 55);
    state[9] = _rotateLeft(state[9], 20);
    state[10] = _rotateLeft(state[10], 3);
    state[11] = _rotateLeft(state[11], 10);
    state[12] = _rotateLeft(state[12], 43);
    state[13] = _rotateLeft(state[13], 25);
    state[14] = _rotateLeft(state[14], 39);
    state[15] = _rotateLeft(state[15], 41);
    state[16] = _rotateLeft(state[16], 45);
    state[17] = _rotateLeft(state[17], 15);
    state[18] = _rotateLeft(state[18], 21);
    state[19] = _rotateLeft(state[19], 8);
    state[20] = _rotateLeft(state[20], 18);
    state[21] = _rotateLeft(state[21], 2);
    state[22] = _rotateLeft(state[22], 61);
    state[23] = _rotateLeft(state[23], 56);
    state[24] = _rotateLeft(state[24], 14);

    keccak_pi(state);
    keccak_chi(state);

    // apply iota(state, round)
    *state ^= keccak_round_constants[round];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief The core transformation. Process the specified block of data.
 *
 * @param hash the algorithm state
 * @param block the message block to process
 * @param block_size the size of the processed block in bytes
 */
void SHA3::
sha3_process_block(uint64_t hash[25], const uint64_t* block, size_t block_size)
{
  // La taille de bloc est au maximum de 144 pour SHA3-224
  // Cela est testé dans keccak_init().

  // expanded loop
  hash[0] ^= le2me_64(block[0]);
  hash[1] ^= le2me_64(block[1]);
  hash[2] ^= le2me_64(block[2]);
  hash[3] ^= le2me_64(block[3]);
  hash[4] ^= le2me_64(block[4]);
  hash[5] ^= le2me_64(block[5]);
  hash[6] ^= le2me_64(block[6]);
  hash[7] ^= le2me_64(block[7]);
  hash[8] ^= le2me_64(block[8]);
  // if not sha3-512
  if (block_size > 72) {
    hash[9] ^= le2me_64(block[9]);
    hash[10] ^= le2me_64(block[10]);
    hash[11] ^= le2me_64(block[11]);
    hash[12] ^= le2me_64(block[12]);
    // if not sha3-384
    if (block_size > 104) {
      hash[13] ^= le2me_64(block[13]);
      hash[14] ^= le2me_64(block[14]);
      hash[15] ^= le2me_64(block[15]);
      hash[16] ^= le2me_64(block[16]);
      // if not sha3-256
      if (block_size > 136) {
        hash[17] ^= le2me_64(block[17]);
      }
    }
  }
  // make a permutation of the hash
  sha3_permutation(hash);
}

#define SHA3_FINALIZED 0x80000000

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calculate message hash.
 *
 * Can be called repeatedly with chunks of the message to be hashed.
 *
 * @param ctx the algorithm context containing current hashing state
 * @param bytes message chunk
 */
void SHA3::
sha3_update(Span<const std::byte> bytes)
{
  sha3_ctx* ctx = &m_context;
  const unsigned char* msg = reinterpret_cast<const unsigned char*>(bytes.data());
  size_t size = bytes.size();

  const size_t index = (size_t)ctx->rest;
  const size_t block_size = (size_t)ctx->block_size;

  if (ctx->rest & SHA3_FINALIZED)
    return; /* too late for additional input */
  ctx->rest = (unsigned)((ctx->rest + size) % block_size);

  /* fill partial block */
  if (index) {
    size_t left = block_size - index;
    std::memcpy((char*)ctx->message + index, msg, (size < left ? size : left));
    if (size < left)
      return;

    /* process partial block */
    sha3_process_block(ctx->hash, ctx->message, block_size);
    msg += left;
    size -= left;
  }
  while (size >= block_size) {
    uint64_t* aligned_message_block;
    if (IS_ALIGNED_64(msg)) {
      /* the most common case is processing of an already aligned message
			without copying it */
      aligned_message_block = (uint64_t*)msg;
    }
    else {
      std::memcpy(ctx->message, msg, block_size);
      aligned_message_block = ctx->message;
    }

    sha3_process_block(ctx->hash, aligned_message_block, block_size);
    msg += block_size;
    size -= block_size;
  }
  if (size) {
    std::memcpy(ctx->message, msg, size); /* save leftovers */
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Store calculated hash into the given array.
 *
 * @param output_hash calculated hash in binary form
 */
void SHA3::
sha3_final(ByteArray& output_hash)
{
  sha3_ctx* ctx = &m_context;
  size_t digest_length = 100 - ctx->block_size / 2;
  const size_t block_size = ctx->block_size;
  output_hash.resize(digest_length);
  auto* result = reinterpret_cast<unsigned char*>(output_hash.data());

  if (!(ctx->rest & SHA3_FINALIZED)) {
    // clear the rest of the data queue
    std::memset((char*)ctx->message + ctx->rest, 0, block_size - ctx->rest);
    ((char*)ctx->message)[ctx->rest] |= 0x06;
    ((char*)ctx->message)[block_size - 1] |= (char)0x80;

    // process final block
    sha3_process_block(ctx->hash, ctx->message, block_size);
    ctx->rest = SHA3_FINALIZED; // mark context as finalized
  }

  if (block_size <= digest_length)
    ARCANE_FATAL("Bad value: block_size={0} digest_length={1}", block_size, digest_length);

  if (result)
    me64_to_le_str(result, ctx->hash, digest_length);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::SHA3Algorithm

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3HashAlgorithm::
_computeHash64(Span<const std::byte> input, ByteArray& output)
{
  using namespace SHA3Algorithm;

  SHA3 sha3;
  this->_initialize(sha3);
  sha3.sha3_update(input);
  sha3.sha3_final(output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3HashAlgorithm::
computeHash(ByteConstArrayView input, ByteArray& output)
{
  Span<const Byte> input64(input);
  return _computeHash64(asBytes(input64), output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3HashAlgorithm::
computeHash64(Span<const Byte> input, ByteArray& output)
{
  Span<const std::byte> bytes(asBytes(input));
  return _computeHash64(bytes, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3HashAlgorithm::
computeHash64(Span<const std::byte> input, ByteArray& output)
{
  return _computeHash64(input, output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3_256HashAlgorithm::
_initialize(SHA3Algorithm::SHA3& sha3)
{
  sha3.sha3_256_init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3_224HashAlgorithm::
_initialize(SHA3Algorithm::SHA3& sha3)
{
  sha3.sha3_224_init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3_384HashAlgorithm::
_initialize(SHA3Algorithm::SHA3& sha3)
{
  sha3.sha3_384_init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SHA3_512HashAlgorithm::
_initialize(SHA3Algorithm::SHA3& sha3)
{
  sha3.sha3_512_init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
