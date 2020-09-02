CREATE database company;
CREATE TABLE tableregister(
    tablenum int not null,
    waiterid int not null,
    menuid int,
    inuse boolean
);
CREATE TABLE plat(
    platid int not null,
    descriptionword varchar(100),
    ingredient varchar(100),
    imageplatpath1 varchar(100),
    imageplatpath2 varchar(100),
    imageplatpath3 varchar(100),
    prohibited varchar(100),
    timecook int,
    primary key(platid)
);
CREATE TABLE tableInfo(
    tablenum int not null,
    reserved boolean,
    occupied boolean,
    primary key(tableNum)
);
CREATE TABLE menu(
    menuid int not null,
    platid int not null,
    ready boolean,
    numbers int
);
