#include "blis.h"
#include <stdio.h>
int main(void){ arch_t id=bli_arch_query_id(); printf("arch=%d string=%s\n",(int)id,bli_arch_string(id)); return 0; }
