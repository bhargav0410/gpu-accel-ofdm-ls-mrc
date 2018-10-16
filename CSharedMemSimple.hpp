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
