/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE

#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "filter.h"
#include "segment-graph.h"

using namespace SN3;
namespace SN3
{
// random color
rgb random_rgb() {
    rgb c;
    double r;

    c.r = (uchar)random();
    c.g = (uchar)random();
    c.b = (uchar)random();

    return c;
}

// dissimilarity measure between pixels
static inline float diff(image<float> *r,int x1, int y1, int x2, int y2) {
    return abs(imRef(r, x1, y1)-imRef(r, x2, y2));
}

/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
image<rgb> *segment_image(image<uchar> *im, float sigma, float c, int min_size,
                          int *num_ccs) {
    int width = im->width();
    int height = im->height();

    image<float> *r = new image<float>(width, height);

    // smooth each color channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            imRef(r, x, y) = (float)imRef(im, x, y);
        }
    }
    image<float> *smooth_r = smooth(r, sigma);
    delete r;

    // build graph
    edge *edges = new edge[width*height*4];
    int num = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width-1) {
                edges[num].a = y * width + x;
                edges[num].b = y * width + (x+1);
                edges[num].w = diff(smooth_r, x, y, x+1, y);
                num++;
            }

            if (y < height-1) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + x;
                edges[num].w = diff(smooth_r, x, y, x, y+1);
                num++;
            }

            if ((x < width-1) && (y < height-1)) {
                edges[num].a = y * width + x;
                edges[num].b = (y+1) * width + (x+1);
                edges[num].w = diff(smooth_r, x, y, x+1, y+1);
                num++;
            }

            if ((x < width-1) && (y > 0)) {
                edges[num].a = y * width + x;
                edges[num].b = (y-1) * width + (x+1);
                edges[num].w = diff(smooth_r, x, y, x+1, y-1);
                num++;
            }
        }
    }
    delete smooth_r;

    // segment
    universe *u = segment_graph(width*height, num, edges, c);

    // post process small components
    for (int i = 0; i < num; i++) {
        int a = u->find(edges[i].a);
        int b = u->find(edges[i].b);
        if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
            u->join(a, b);
    }
    delete [] edges;
    *num_ccs = u->num_sets();

    image<rgb> *output = new image<rgb>(width, height);

    int m=0, n=0;
    int lightArr[*num_ccs];
    rgb color;

    for(int y=0; y<*num_ccs; y++) {
        lightArr[y] = 0;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int comp = u->find(y * width + x);
            for(m=0; m<n; m++) {
                if(comp == lightArr[m]) {
                    color.r = m/255;
                    color.g = m%255;
                    color.b = 0;
                    imRef(output, x, y) = color;
                    break;
                }
            }
            if(m == n) {
                color.r = m/255;
                color.g = m%255;
                color.b = 0;
                imRef(output, x, y) = color;
                lightArr[m] = comp;
                n += 1;
            }
        }
    }
    delete u;

    return output;
}

}
#endif
