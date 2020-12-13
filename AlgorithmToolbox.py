from shapely.geometry import Point, Polygon

def within_triangle(low, high, queryPoint):
    origin = Point([0,0,0])
    p1 = Point(low)
    p2 = Point(high)
    coords = [origin, p1, p2]
    poly = Polygon(coords)
    query = Point(queryPoint)
    return query.within(poly)

if __name__ == '__main__':
    # Create Point objects
    p1 = Point([24.952242, 60.1696017, 1])
    p2 = Point([24.976567, 60.1612500, 1])
    p3 = Point([-1,-1,-1])

    # Create a Polygon
    coords = [[24.950899, 60.169158, 1], [24.953492, 60.169158, 1], [24.953510, 60.170104, 1], [24.950958, 60.169990, 1]]
    poly = Polygon(coords)

    print(p3.within(poly))