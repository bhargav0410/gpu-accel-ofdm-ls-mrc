/*
Copyright (c) 2018, WINLAB, Rutgers University, USA
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _CSHAREDMEMSIMPLE_HPP_
#define _CSHAREDMEMSIMPLE_HPP_

#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#if HAVE_UNISTD_H
	#include <unistd.h>
#else
	#include <io.h>
#endif
#include <sys/mman.h>
#include <string>

#define PAGE_SZ (256*8) // 256 from g_spb, 8 from size of std:complex<float>


struct s_page_fmt {
  unsigned int nBytes;
  unsigned int seqno;
  char buf[PAGE_SZ];
};

#define NUM_PAGES (32)
struct s_notebook {
  struct s_page_fmt pg[NUM_PAGES];
};


struct __s_page {
  unsigned int nBytes;
  unsigned int seqno;
  unsigned int new_data;
  long unsigned int clock_tick;
  unsigned int buf_offset;
};

struct __s_ntbk {
  struct __s_page pg[NUM_PAGES];
  char buf[PAGE_SZ * NUM_PAGES];
};

class CSharedMemSimple
{
private:
  void *rptr_;
  int fd_;
  bool isMaster_;
  std::string shm_uid_;
  unsigned int bytes_allocated_;
  void handle_error(const char *msg)
  {
    do { 
      perror(msg);
      exit(EXIT_FAILURE);
    } while (0);
  }

public:
  // Constructor - create if necessary and set to RW mode
  CSharedMemSimple(std::string shm_uid, unsigned int sizeInBytes)
  {
    isMaster_ = false;
    shm_uid_ = shm_uid;
    bytes_allocated_ = sizeInBytes;

    /* Create shared memory object and set its size */
    if ((fd_ = shm_open(shm_uid_.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) == -1)
      handle_error("open");

    if (ftruncate(fd_, bytes_allocated_) == -1)
      handle_error("ftruncate");

    /* Map shared memory object */
    rptr_ = (void *) mmap(NULL, bytes_allocated_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (rptr_ == MAP_FAILED)
      exit(-1); /* Handle error */
  }


  // Destructor - releases shared memory
  ~CSharedMemSimple()
  {
	if (isMaster_ == false)
      return;

    munmap(rptr_, bytes_allocated_);
    shm_unlink(shm_uid_.c_str());
  }

  void set_master_mode()
  {
    isMaster_ = true;
  }

  unsigned int nBytes()
  {
    return (bytes_allocated_);
  }

  void* ptr()
  {
    return (void *)(rptr_);
  }

  void info()
  {
    printf("SHM info: %s, %s\n", shm_uid_.c_str(), (isMaster_) ? "Master" : "Slave"  );
    printf("SHM bytes allocated: %i\n", nBytes()  );
  }


};


#endif
